import argparse
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.canbus.canbus_pb2 import Twist2d


def make_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


async def follow(
    canbus_cfg_path: Path,
    camera_cfg_path: Path,
    *,
    linear_speed: float,
    max_angular: float,
    deadband_px_frac: float,
    kp_angular: float,
    lost_timeout_s: float,
    send_hz: float,
):
    canbus_cfg: EventServiceConfig = proto_from_json_file(canbus_cfg_path, EventServiceConfig())
    cam_cfg: EventServiceConfig = proto_from_json_file(camera_cfg_path, EventServiceConfig())

    canbus_client = EventClient(canbus_cfg)
    cam_client = EventClient(cam_cfg)

    hog = make_hog()

    latest = {"frame": None, "cx": None, "ts": 0.0}

    async def camera_loop():
        sub = cam_cfg.subscriptions[0]
        async for _event, msg in cam_client.subscribe(sub, decode=True):
            frame = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Speed up detection a bit
            small = cv2.resize(frame, None, fx=0.5, fy=0.5)
            rects, weights = hog.detectMultiScale(
                small,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )

            cx = None
            if len(rects) > 0:
                # Choose the best candidate (highest weight if available)
                best_i = int(np.argmax(weights)) if len(weights) == len(rects) and len(weights) > 0 else 0
                x, y, w, h = rects[best_i]

                # Scale bbox back up
                x, y, w, h = [int(v * 2) for v in (x, y, w, h)]
                cx = x + w // 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, y + h // 2), 6, (0, 0, 255), -1)

            latest["frame"] = frame
            latest["cx"] = cx
            latest["ts"] = time.time()

    async def control_loop():
        period = 1.0 / send_hz
        last_sent = 0.0

        while True:
            now = time.time()
            frame = latest["frame"]
            cx = latest["cx"]
            age = now - latest["ts"]

            twist = Twist2d()
            twist.linear_velocity_x = 0.0
            twist.angular_velocity = 0.0

            if frame is not None:
                h, w = frame.shape[:2]
                center = w // 2
                deadband_px = int(w * deadband_px_frac)

                if cx is not None and age <= lost_timeout_s:
                    err = center - cx
                    if abs(err) <= deadband_px:
                        twist.linear_velocity_x = float(linear_speed)
                        twist.angular_velocity = 0.0
                    else:
                        ang = float(np.clip(kp_angular * (err / center), -max_angular, max_angular))
                        twist.linear_velocity_x = 0.0
                        twist.angular_velocity = ang

                status = f"cx={cx} age={age:.2f}s lin={twist.linear_velocity_x:.2f} ang={twist.angular_velocity:.2f}"
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Follow Person (HOG)", frame)

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
    ap.add_argument("--linear", type=float, default=0.20)
    ap.add_argument("--max-angular", type=float, default=0.8)
    ap.add_argument("--deadband", type=float, default=0.10)
    ap.add_argument("--kp", type=float, default=1.2)
    ap.add_argument("--lost-timeout", type=float, default=0.5)
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    asyncio.run(
        follow(
            args.canbus_config,
            args.camera_config,
            linear_speed=args.linear,
            max_angular=args.max_angular,
            deadband_px_frac=args.deadband,
            kp_angular=args.kp,
            lost_timeout_s=args.lost_timeout,
            send_hz=args.hz,
        )
    )