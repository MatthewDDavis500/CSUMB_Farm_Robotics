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

async def follow(
    canbus_cfg_path: Path,
    camera_cfg_path: Path,
    *,
    model_name: str,
    conf: float,
    iou: float,
    infer_scale: float,          # downscale factor for YOLO (0.5 = half-res)
    ema_alpha: float,            # smoothing for cx (0..1), higher = smoother
    linear_max_fwd: float,
    linear_max_rev: float,
    target_h_frac: float,        # desired bbox height fraction (proxy distance)
    h_deadband: float,           # +/- around target where we command 0 linear
    kp_linear: float,            # proportional gain for forward/back based on h_frac error
    max_angular: float,
    deadband_px_frac: float,
    kp_angular: float,
    lost_timeout_s: float,
    send_hz: float,
    flip_steer: bool,
):
    canbus_cfg: EventServiceConfig = proto_from_json_file(canbus_cfg_path, EventServiceConfig())
    cam_cfg: EventServiceConfig = proto_from_json_file(camera_cfg_path, EventServiceConfig())

    canbus_client = EventClient(canbus_cfg)
    cam_client = EventClient(cam_cfg)

    model = YOLO(model_name)

    latest = {
        "frame": None,
        "cx": None,
        "cx_smooth": None,
        "ts": 0.0,
        "box": None,       # (x1,y1,x2,y2) full-res
        "score": None
    }

    async def camera_loop():
        sub = cam_cfg.subscriptions[0]

        async for _event, msg in cam_client.subscribe(sub, decode=True):
            frame = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Optional downscale for faster inference
            if infer_scale != 1.0:
                small = cv2.resize(frame, None, fx=infer_scale, fy=infer_scale)
            else:
                small = frame

            # YOLO inference; class 0 is "person" for COCO models
            results = model.predict(
                source=small,
                conf=conf,
                iou=iou,
                classes=[0],
                verbose=False,
            )

            cx = None
            best_box = None
            best_score = None

            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # (N,4)
                    scores = r.boxes.conf.cpu().numpy()       # (N,)

                    # Closest person = largest bbox area
                    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                    idx = int(np.argmax(areas))

                    x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
                    best_score = float(scores[idx])

                    # Scale to full-res if downscaled
                    if infer_scale != 1.0:
                        x1 /= infer_scale; y1 /= infer_scale
                        x2 /= infer_scale; y2 /= infer_scale

                    x1 = int(clamp(x1, 0, w - 1))
                    y1 = int(clamp(y1, 0, h - 1))
                    x2 = int(clamp(x2, 0, w - 1))
                    y2 = int(clamp(y2, 0, h - 1))

                    best_box = (x1, y1, x2, y2)
                    cx = (x1 + x2) // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, (y1 + y2) // 2), 6, (0, 0, 255), -1)
                    cv2.putText(frame, f"person conf={best_score:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Smooth cx (EMA)
            cx_smooth = latest["cx_smooth"]
            if cx is not None:
                if cx_smooth is None:
                    cx_smooth = float(cx)
                else:
                    cx_smooth = (ema_alpha * cx_smooth) + ((1.0 - ema_alpha) * float(cx))

            latest["frame"] = frame
            latest["cx"] = cx
            latest["cx_smooth"] = cx_smooth
            latest["box"] = best_box
            latest["score"] = best_score
            latest["ts"] = time.time()

    async def control_loop():  #;lkhgfdsdfgjkl
        period = 1.0 / send_hz
        last_sent = 0.0

        while True:
            now = time.time()
            frame = latest["frame"]
            cx_smooth = latest["cx_smooth"]
            box = latest["box"]
            score = latest["score"]
            age = now - latest["ts"]

            twist = Twist2d()
            twist.linear_velocity_x = 0.0
            twist.angular_velocity = 0.0

            if frame is not None:
                h, w = frame.shape[:2]
                center = w // 2
                deadband_px = int(w * deadband_px_frac)

                # Default: stop if we don't have a fresh person
                have_target = (box is not None) and (cx_smooth is not None) and (age <= lost_timeout_s)

                if have_target:
                    x1, y1, x2, y2 = box
                    box_h = max(1, (y2 - y1))
                    h_frac = box_h / float(h)

                    # --- Distance control (forward/back) ---
                    # error > 0 means person is far (bbox small) => move forward
                    # error < 0 means person is close (bbox large) => move backward
                    dist_err = target_h_frac - h_frac

                    if abs(dist_err) <= h_deadband:
                        lin_cmd = 0.0
                    else:
                        lin_cmd = kp_linear * dist_err  # proportional
                        # clamp forward / reverse separately
                        lin_cmd = clamp(lin_cmd, -linear_max_rev, linear_max_fwd)

                    # --- Heading control (left/right) ---
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

                    # Debug overlay
                    cv2.line(frame, (center, 0), (center, h), (255, 255, 0), 2)
                    cv2.putText(frame, f"h_frac={h_frac:.2f} target={target_h_frac:.2f} err={dist_err:.2f}",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"lin={lin_cmd:.2f} ang={ang_cmd:.2f} conf={0.0 if score is None else score:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, f"LOST target age={age:.2f}s (stopping)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Follow Person (YOLO Distance)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    await canbus_client.request_reply("/twist", Twist2d())
                    return

            if now - last_sent >= period:
                await canbus_client.request_reply("/twist", twist)
                last_sent = now

            await asyncio.sleep(0.001)

    cam_task = asyncio.create_task(camera_loop())
    try:
        await control_loop()
    finally:
        cam_task.cancel()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canbus-config", type=Path, required=True)
    ap.add_argument("--camera-config", type=Path, required=True)

    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.60, help="Raise to reduce false positives")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--infer-scale", type=float, default=0.6, help="0.5-1.0; lower=faster")

    ap.add_argument("--ema", type=float, default=0.85, help="cx smoothing (0..1)")

    # Distance control settings (bbox-height proxy)
    ap.add_argument("--target-h", type=float, default=0.50, help="Desired bbox height fraction (0..1)")
    ap.add_argument("--h-deadband", type=float, default=0.05, help="+/- around target where we stop")
    ap.add_argument("--kp-linear", type=float, default=0.8, help="Linear proportional gain")
    ap.add_argument("--max-fwd", type=float, default=0.20, help="Max forward speed")
    ap.add_argument("--max-rev", type=float, default=0.15, help="Max reverse speed")

    # Heading control
    ap.add_argument("--max-angular", type=float, default=0.8)
    ap.add_argument("--deadband", type=float, default=0.10)
    ap.add_argument("--kp-ang", type=float, default=1.2)
    ap.add_argument("--flip-steer", action="store_true", help="Flip turning direction if needed")

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
            linear_max_fwd=args.max_fwd,
            linear_max_rev=args.max_rev,
            target_h_frac=args.target_h,
            h_deadband=args.h_deadband,
            kp_linear=args.kp_linear,
            max_angular=args.max_angular,
            deadband_px_frac=args.deadband,
            kp_angular=args.kp_ang,
            lost_timeout_s=args.lost_timeout,
            send_hz=args.hz,
            flip_steer=args.flip_steer,
        )
    )