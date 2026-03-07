import argparse
import asyncio
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore

from farm_ng.core.event_client import EventClient  # type: ignore
from farm_ng.core.event_service_pb2 import EventServiceConfig  # type: ignore
from farm_ng.core.events_file_reader import proto_from_json_file # type: ignore
from farm_ng.canbus.canbus_pb2 import Twist2d  # type: ignore

### Config File Paths ###
OAK_CONFIG = 'oak_config.json'
CANBUS_CONFIG = 'canbus_config.json'

### Model Parameters ###
MODEL_NAME = 'yolov8n.pt'    # (String) YOLO file to load
CONFIDENCE_THRESHOLD = 0.65  # (0-1 Scale) How confident does the model need to be to consider an object a person?
IOU = 0.5                    # (0-1 Scale) Intersection over union. How much do two bounding boxes need to overlap for the model to consider them the same object and only take the one with the heighest confindence?
FRAME_SCALING = 0.6          # (Proportional Multiplier) Ratio by which to scale the camera frames before passing to model. Smaller numbers means smaller image size, which means faster computation but less accuracy
EMA_ALPHA = 0.85             # (0-1 Scale) Smoothness of reaction. Higher values means more memory (previous frames) is considered, so smoother but slower reactions. Lower values mean more new frames are considered, resulting in faster but possibly chattery responses

### Distance Control Parameters ###
TARGET_HEIGHT = 0.8         # (Fraction) What fraction of the frame vertically should the detected person be ideally filling? Higher values means closer following
HEIGHT_DEADZONE = 0.05      # (Fraction) Deadzone for distance control. Having a detected person filling this much more or less than the TARGET_HEIGHT will be acceptable
KP_LINEAR = 0.8             # (Gain) Proportion for how fast the robot will move forward or backward based on the current error
MAX_FORWARD = 0.25          # (Meters/Second) Maximum forward velocity
MAX_REVERSE = 0.18          # (Meters/Second) Maximum reverse velocity

### Heading Control Parameters ###
FLIP_STEER = True           # (Boolean) If True, inverts steering
KP_ANGULAR = 1.2            # (Gain) Proportion for how fast the robot will turn based on the current error
MAX_ANGULAR = 0.5           # (Radians/Second) Maximum angular velocity
ANGULAR_DEADZONE = 0.1      # (Fraction) Deadzone for turning. Having a detected person within this much of the center will be acceptable

### Safety/Performance ###
LOST_TIMEOUT = 0.8          # (Seconds) How long the robot will continue movement before stopping completely if it doesn't detect anyone
SEND_HZ = 20.0              # (Hertz: X/Second) How many times per second twist commands will be sent to the motors


def clamp(value: float, min: float, max: float) -> float:
    '''
    Constrains a float value to be within a defined minimum and maximum range.
    
    Used here to saturate control signals (like speed and steering) so they 
    do not exceed the physical or safety limits of the robot.
    
    Args:
        value: The input value to be constrained.
        min: The lower bound (minimum allowed value).
        max: The upper bound (maximum allowed value).
        
    Returns:
        The constrained float value:
            If value is within limits, returns value.
            If value is above max, returns max.
            If value is below min, returns min.
    '''
    temp = min(max, value)
    return float(max(min, temp))


async def follow():
    canbus_cfg: EventServiceConfig = proto_from_json_file(CANBUS_CONFIG, EventServiceConfig())
    cam_cfg: EventServiceConfig = proto_from_json_file(OAK_CONFIG, EventServiceConfig())

    canbus_client = EventClient(canbus_cfg)
    cam_client = EventClient(cam_cfg)

    if len(cam_cfg.subscriptions) < 2:
        raise ValueError("camera config must contain 2 subscriptions (oak0 + oak1)")

    model = YOLO(MODEL_NAME)

    # Per-camera latest state
    states = {
        'oak0': {
            "frame": None,            # Image from the Oak camera to be altered and displayed by OpenCV
            "target_center_x": None,  # Target person's horizontal center, smoothed to avoid motor jittering
            "box": None,              # Bounding box dimentions in full-res: (x1,y1,x2,y2)
            "height_fraction": None,  # Bounding box height fraction of frame
            "score": None,            # Model confidence in person detection
            "timestamp": 0.0,         # Timestamp of current state
        },
        'oak1': {
            "frame": None,
            "target_center_x": None,
            "box": None,
            "height_fraction": None,
            "score": None,
            "timestamp": 0.0,
        }
    }

    async def camera_worker(sub, cam_name):
        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)

        async for event, msg in cam_client.subscribe(sub, decode=True):
            frame = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Optional downscale to speed up YOLO
            if FRAME_SCALING != 1.0:
                small = cv2.resize(frame, None, fx=FRAME_SCALING, fy=FRAME_SCALING)
            else:
                small = frame

            # Detect people only (COCO class 0)
            results = model.predict(
                source=small,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU,
                classes=[0],
                verbose=False,
            )

            best_box = None
            best_score = None
            cx = None
            height_fraction = None

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
                    if FRAME_SCALING != 1.0:
                        x1 /= FRAME_SCALING; y1 /= FRAME_SCALING
                        x2 /= FRAME_SCALING; y2 /= FRAME_SCALING

                    x1 = int(clamp(x1, 0, w - 1))
                    y1 = int(clamp(y1, 0, h - 1))
                    x2 = int(clamp(x2, 0, w - 1))
                    y2 = int(clamp(y2, 0, h - 1))

                    best_box = (x1, y1, x2, y2)
                    cx = (x1 + x2) // 2

                    box_h = max(1, (y2 - y1))
                    height_fraction = box_h / float(h)

                    # Draw bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, (y1 + y2) // 2), 6, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"{cam_name} conf={best_score:.2f} h={height_fraction:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            # EMA smoothing for cx
            target_center_x = states[cam_name]["target_center_x"]
            if cx is not None:
                if target_center_x is None:
                    target_center_x = float(cx)
                else:
                    target_center_x = (EMA_ALPHA * target_center_x) + ((1.0 - EMA_ALPHA) * float(cx))

            states[cam_name]["frame"] = frame
            states[cam_name]["target_center_x"] = target_center_x
            states[cam_name]["box"] = best_box
            states[cam_name]["height_fraction"] = height_fraction
            states[cam_name]["score"] = best_score
            states[cam_name]["timestamp"] = time.time()

            # show each camera stream
            cv2.imshow(cam_name, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                # user quits: just return and let main exit
                return

    async def control_loop():
        period = 1.0 / SEND_HZ
        last_sent = 0.0

        cv2.namedWindow("ACTIVE_TARGET", cv2.WINDOW_NORMAL)

        while True:
            now = time.time()

            # Choose the best/closest target across cameras
            best_cam = None
            best_h = -1.0

            for cam_name, st in states.items():
                age = now - st["timestamp"]
                if age > LOST_TIMEOUT:
                    continue
                if st["box"] is None or st["target_center_x"] is None or st["height_fraction"] is None:
                    continue

                # "Closest overall" = largest bbox height fraction
                if st["height_fraction"] > best_h:
                    best_h = st["height_fraction"]
                    best_cam = cam_name

            twist = Twist2d()
            twist.linear_velocity_x = 0.0
            twist.angular_velocity = 0.0

            if best_cam is not None:
                st = states[best_cam]
                frame = st["frame"]
                target_center_x = st["target_center_x"]
                box = st["box"]
                height_fraction = st["height_fraction"]
                score = st["score"]

                h, w = frame.shape[:2]
                center = w // 2
                deadband_px = int(w * ANGULAR_DEADZONE)

                # Distance control: target_height_fraction (your requested 0.80 default)
                dist_err = TARGET_HEIGHT - float(height_fraction)  # >0 => too far => forward; <0 => too close => reverse

                if abs(dist_err) <= HEIGHT_DEADZONE:
                    lin_cmd = 0.0
                else:
                    lin_cmd = KP_LINEAR * dist_err
                    lin_cmd = clamp(lin_cmd, -MAX_REVERSE, MAX_FORWARD)

                # Heading control (flipped by default per your request)
                steer_err = float(target_center_x - center)
                if FLIP_STEER:
                    steer_err = -steer_err

                if abs(steer_err) <= deadband_px:
                    ang_cmd = 0.0
                else:
                    ang_cmd = KP_ANGULAR * (steer_err / center)
                    ang_cmd = clamp(ang_cmd, -MAX_ANGULAR, MAX_ANGULAR)

                twist.linear_velocity_x = float(lin_cmd)
                twist.angular_velocity = float(ang_cmd)

                # Active target debug window
                active = frame.copy()
                cv2.line(active, (center, 0), (center, h), (255, 255, 0), 2)
                cv2.putText(active, f"ACTIVE: {best_cam}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(active, f"height_fraction={height_fraction:.2f} target={TARGET_HEIGHT:.2f} lin={lin_cmd:.2f} ang={ang_cmd:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(active, f"conf={0.0 if score is None else score:.2f} flip_steer={FLIP_STEER}",
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
    cam_tasks = []
    name = ''
    for sub in cam_cfg.subscriptions:
        # Determine camera name
        if 'oak0' in sub['uri']['query']:
            name = 'oak0'
        elif 'oak1' in sub['uri']['query']:
            name = 'oak1'
        else:
            name = 'camera'
            
        # Add asynchronous task to task list
        cam_tasks.append(asyncio.create_task(camera_worker(sub, name)))
        
    try:
        await control_loop()
    finally:
        for t in cam_tasks:
            t.cancel()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    asyncio.run(
        follow()
    )

## Alex's Hotspot Amiga IP: 10.179.67.170
## Matthew's Hotspot Amiga IP: 172.20.10.4