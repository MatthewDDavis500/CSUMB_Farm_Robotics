import argparse
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.canbus.canbus_pb2 import Twist2d


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def parse_service_name(query: str) -> str:
    # query like: "service_name=oak0"
    if not query:
        return "camera"
    parts = {}
    for kv in query.split("&"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            parts[k] = v
    return parts.get("service_name", "camera")


async def follow(
    canbus_cfg_path: Path,
    camera_cfg_path: Path,
    *,
    model_name: str,
    conf: float,
    iou: float,
    infer_scale: float,
    ema_alpha: float,
    # Distance control
    target_h_frac: float,
    h_deadband: float,
    kp_linear: float,
    linear_max_fwd: float,
    linear_max_rev: float,
    # Heading control
    flip_steer: bool,
    kp_angular: float,
    max_angular: float,
    deadband_px_frac: float,
    # General
    lost_timeout_s: float,
    send_hz: float,
):
    canbus_cfg: EventServiceConfig = proto_from_json_file(canbus_cfg_path, EventServiceConfig())
    cam_cfg: EventServiceConfig = proto_from_json_file(camera_cfg_path, EventServiceConfig())

    canbus_client = EventClient(canbus_cfg)
    cam_client = EventClient(cam_cfg)

    if len(cam_cfg.subscriptions) < 2:
        raise ValueError("camera config must contain 2 subscriptions (oak0 + oak1)")

    model = YOLO(model_name)

    # Per-camera latest state
    # Each entry stores: frame, cx_smooth, bbox, h_frac, score, ts
    states = {}
    for sub in cam_cfg.subscriptions:
        cam_name = parse_service_name(sub.uri.query)
        states[cam_name] = {
            "frame": None,
            "cx_smooth": None,
            "box": None,      # (x1,y1,x2,y2) in full-res
            "h_frac": None,   # bbox height fraction of frame
            "score": None,
            "ts": 0.0,
        }

    async def camera_worker(sub):
        cam_name = parse_service_name(sub.uri.query)
        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)

        async for event, msg in cam_client.subscribe(sub, decode=True):
            frame = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Optional downscale to speed up YOLO
            if infer_scale != 1.0:
                small = cv2.resize(frame, None, fx=infer_scale, fy=infer_scale)
            else:
                small = frame

            # Detect people only (COCO class 0)
            results = model.predict(
                source=small,
                conf=conf,
                iou=iou,
                classes=[0],
                verbose=False,
            )

            best_box = None
            best_score = None
            cx = None
            h_frac = None

            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()

                    # Closest in THIS camera = largest bbox area
                    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                    idx = int(np.argmax(areas))

                    x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
                    best_score = float(scores[idx])

                    # Scale to full-res
                    if infer_scale != 1.0:
                        x1 /= infer_scale; y1 /= infer_scale
                        x2 /= infer_scale; y2 /= infer_scale

                    x1 = int(clamp(x1, 0, w - 1))
                    y1 = int(clamp(y1, 0, h - 1))
                    x2 = int(clamp(x2, 0, w - 1))
                    y2 = int(clamp(y2, 0, h - 1))

                    best_box = (x1, y1, x2, y2)
                    cx = (x1 + x2) // 2

                    box_h = max(1, (y2 - y1))
                    h_frac = box_h / float(h)

                    # Draw bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, (y1 + y2) // 2), 6, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"{cam_name} conf={best_score:.2f} h={h_frac:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            # EMA smoothing for cx
            cx_smooth = states[cam_name]["cx_smooth"]
            if cx is not None:
                if cx_smooth is None:
                    cx_smooth = float(cx)
                else:
                    cx_smooth = (ema_alpha * cx_smooth) + ((1.0 - ema_alpha) * float(cx))

            states[cam_name]["frame"] = frame
            states[cam_name]["cx_smooth"] = cx_smooth
            states[cam_name]["box"] = best_box
            states[cam_name]["h_frac"] = h_frac
            states[cam_name]["score"] = best_score
            states[cam_name]["ts"] = time.time()

            # show each camera stream
            cv2.imshow(cam_name, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                # user quits: just return and let main exit
                return

    async def control_loop():
        period = 1.0 / send_hz
        last_sent = 0.0

        cv2.namedWindow("ACTIVE_TARGET", cv2.WINDOW_NORMAL)

        while True:
            now = time.time()

            # Choose the best/closest target across cameras
            best_cam = None
            best_h = -1.0

            for cam_name, st in states.items():
                age = now - st["ts"]
                if age > lost_timeout_s:
                    continue
                if st["box"] is None or st["cx_smooth"] is None or st["h_frac"] is None:
                    continue

                # "Closest overall" = largest bbox height fraction
                if st["h_frac"] > best_h:
                    best_h = st["h_frac"]
                    best_cam = cam_name

            twist = Twist2d()
            twist.linear_velocity_x = 0.0
            twist.angular_velocity = 0.0

            if best_cam is not None:
                st = states[best_cam]
                frame = st["frame"]
                cx_smooth = st["cx_smooth"]
                box = st["box"]
                h_frac = st["h_frac"]
                score = st["score"]

                h, w = frame.shape[:2]
                center = w // 2
                deadband_px = int(w * deadband_px_frac)

                # Distance control: target_h_frac (your requested 0.80 default)
                dist_err = target_h_frac - float(h_frac)  # >0 => too far => forward; <0 => too close => reverse

                if abs(dist_err) <= h_deadband:
                    lin_cmd = 0.0
                else:
                    lin_cmd = kp_linear * dist_err
                    lin_cmd = clamp(lin_cmd, -linear_max_rev, linear_max_fwd)

                # Heading control (flipped by default per your request)
                steer_err = float(cx_smooth - center)
                if flip_steer:
                    steer_err = -steer_err

                if abs(steer_err) <= deadband_px:
                    ang_cmd = 0.0
                else:
                    ang_cmd = kp_angular * (steer_err / center)
                    ang_cmd = clamp(ang_cmd, -max_angular, max_angular)

                twist.linear_velocity_x = float(lin_cmd)
                twist.angular_velocity = float(ang_cmd)

                # Active target debug window
                active = frame.copy()
                cv2.line(active, (center, 0), (center, h), (255, 255, 0), 2)
                cv2.putText(active, f"ACTIVE: {best_cam}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(active, f"h_frac={h_frac:.2f} target={target_h_frac:.2f} lin={lin_cmd:.2f} ang={ang_cmd:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(active, f"conf={0.0 if score is None else score:.2f} flip_steer={flip_steer}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("ACTIVE_TARGET", active)
            else:
                # No target => stop
                blank = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "NO TARGET (stopping)", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("ACTIVE_TARGET", blank)

            # Quit if q pressed in any window
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                await canbus_client.request_reply("/twist", Twist2d())
                return

            # Publish at fixed rate
            if now - last_sent >= period:
                await canbus_client.request_reply("/twist", twist)
                last_sent = now

            await asyncio.sleep(0.001)

    # Run camera workers (2 cameras) + control loop
    cam_tasks = [asyncio.create_task(camera_worker(sub)) for sub in cam_cfg.subscriptions]
    try:
        await control_loop()
    finally:
        for t in cam_tasks:
            t.cancel()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canbus-config", type=Path, required=True)
    ap.add_argument("--camera-config", type=Path, required=True)

    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--infer-scale", type=float, default=0.6)
    ap.add_argument("--ema", type=float, default=0.85)

    # Your requested defaults:
    ap.add_argument("--target-h", type=float, default=0.80)   # distance setpoint
    ap.add_argument("--h-deadband", type=float, default=0.05)
    ap.add_argument("--kp-linear", type=float, default=0.8)
    ap.add_argument("--max-fwd", type=float, default=0.25)    # slightly up
    ap.add_argument("--max-rev", type=float, default=0.18)    # slightly up

    ap.add_argument("--flip-steer", action="store_true", default=True)  # flipped by default
    ap.add_argument("--kp-ang", type=float, default=1.2)
    ap.add_argument("--max-angular", type=float, default=0.8)
    ap.add_argument("--deadband", type=float, default=0.10)

    ap.add_argument("--lost-timeout", type=float, default=0.8)
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    asyncio.run(
        follow(
            args.canbus_config,
            args.camera_config,
            model_name=args.model,
            conf=args.conf,
            iou=args.iou,
            infer_scale=args.infer_scale,
            ema_alpha=args.ema,
            target_h_frac=args.target_h,
            h_deadband=args.h_deadband,
            kp_linear=args.kp_linear,
            linear_max_fwd=args.max_fwd,
            linear_max_rev=args.max_rev,
            flip_steer=args.flip_steer,
            kp_angular=args.kp_ang,
            max_angular=args.max_angular,
            deadband_px_frac=args.deadband,
            lost_timeout_s=args.lost_timeout,
            send_hz=args.hz,
        )
    )