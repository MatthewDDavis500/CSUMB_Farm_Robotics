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

from farm_ng.oak import oak_pb2
from google.protobuf.empty_pb2 import Empty

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

### Config File Paths ###
OAK_CONFIG = str(ROOT_DIR / "CSUMB_Farm_Robotics" / "config" / "oak_config.json")
CANBUS_CONFIG = str(ROOT_DIR / "CSUMB_Farm_Robotics" / "config" / "canbus_config.json")

### Model Parameters ###
MODEL_NAME = str(ROOT_DIR / "data" / "models" / "yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.65
IOU = 0.5
FRAME_SCALING = 0.6
EMA_ALPHA = 0.85

### Detection Box Vertex Indicies ###
X1 = 0
Y1 = 1
X2 = 2
Y2 = 3

### Distance Control Parameters (DEPTH BASED) ###
TARGET_Z_M = 1.5          # desired follow distance in meters
Z_DEADZONE_M = 0.15       # meters
KP_LINEAR_Z = 0.7         # linear gain
MAX_FORWARD = 0.4
MAX_REVERSE = 0.3

### Heading Control Parameters ###
FLIP_STEER = True
KP_ANGULAR = 1.2
MAX_ANGULAR = 0.5
ANGULAR_DEADZONE = 0.1

### Safety/Performance ###
LOST_TIMEOUT = 0.8
SEND_HZ = 20.0


def clamp(value: float, min_val: float, max_val: float) -> float:
    temp = min(max_val, value)
    return float(max(min_val, temp))


def median_depth_from_disparity(
    disp: np.ndarray,
    box: tuple[int, int, int, int],
    f_px: float,
    baseline_m: float,
    disp_scale: float,
) -> float | None:
    """
    Estimate depth (meters) for a detection by sampling disparity inside the bbox.

    Z = (f * B) / d

    disp: decoded disparity image (often uint16)
    box: (x1,y1,x2,y2) in same pixel space as disp (we run YOLO on /left for this reason)
    f_px: focal length in pixels (from /calibration)
    baseline_m: stereo baseline in meters
    disp_scale: how disparity is scaled in the image (common: 16.0). If your Z is ~16x off, change this.
    """
    if disp is None or box is None:
        return None
    if f_px <= 0 or baseline_m <= 0 or disp_scale <= 0:
        return None

    x1, y1, x2, y2 = box
    H, W = disp.shape[:2]

    # Clamp bbox to disparity image bounds
    x1 = int(clamp(x1, 0, W - 1))
    x2 = int(clamp(x2, 0, W))
    y1 = int(clamp(y1, 0, H - 1))
    y2 = int(clamp(y2, 0, H))

    if x2 <= x1 or y2 <= y1:
        return None

    # Use a center ROI to avoid edges/background (more stable)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    rx1 = x1 + int(0.30 * bw)
    rx2 = x2 - int(0.30 * bw)
    ry1 = y1 + int(0.30 * bh)
    ry2 = y2 - int(0.10 * bh)

    rx1 = int(clamp(rx1, 0, W - 1))
    rx2 = int(clamp(rx2, 0, W))
    ry1 = int(clamp(ry1, 0, H - 1))
    ry2 = int(clamp(ry2, 0, H))

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = disp[ry1:ry2, rx1:rx2].astype(np.float32)

    # If disparity arrives as HxWxC, reduce to one channel
    if roi.ndim == 3:
        roi = roi[..., 0]

    d = roi / float(disp_scale)

    # Filter invalid
    d = d[np.isfinite(d)]
    d = d[d > 0.5]  # discard near-zero disparity
    if d.size < 100:
        return None

    d_med = float(np.median(d))
    if d_med <= 0:
        return None

    z = (float(f_px) * float(baseline_m)) / d_med

    # sanity clamp
    if z < 0.2 or z > 20.0:
        return None
    return z


async def follow(ip: str, baseline_m: float, disp_scale: float, target_z_m: float):
    # Import configuration from JSON config files
    canbus_config: EventServiceConfig = proto_from_json_file(CANBUS_CONFIG, EventServiceConfig())
    oak_config: EventServiceConfig = proto_from_json_file(OAK_CONFIG, EventServiceConfig())

    # Load user-provided IP into config
    canbus_config.host = ip
    oak_config.host = ip

    # Create clients for the canbus and oak cameras
    canbus_client = EventClient(canbus_config)
    oak_client = EventClient(oak_config)

    # Request calibration ONCE and extract focal length (pixels)
    # OakCalibration contains camera_data[] with intrinsic_matrix flattened 3x3.
    # fx is intrinsic_matrix[0].
    try:
        calibration: oak_pb2.OakCalibration = await oak_client.request_reply("/calibration", Empty(), decode=True)
        if len(calibration.camera_data) == 0:
            raise RuntimeError("OakCalibration.camera_data is empty")
        fx = float(calibration.camera_data[0].intrinsic_matrix[0])
        if fx <= 0:
            raise RuntimeError("Calibration fx <= 0")
        F_PX = fx
        print(f"[CALIB] fx={F_PX:.2f}px (from /calibration), baseline={baseline_m:.3f}m, disp_scale={disp_scale}")
    except Exception as e:
        # Fallback so the script still runs, but Z will be wrong until calibration works
        F_PX = 400.0
        print(f"[WARN] Failed to read /calibration ({e}). Using fallback fx={F_PX:.2f}px.")

    # Apply target distance
    global TARGET_Z_M
    TARGET_Z_M = float(target_z_m)

    # Load the YOLO model
    model = YOLO(MODEL_NAME)
    inference_lock = asyncio.Lock()

    # Per-camera latest state
    states = {
        "oak0": {
            "frame": None,              # left frame for display
            "target_center_x": None,
            "box": None,
            "score": None,
            "timestamp": 0.0,

            "disp": None,               # disparity frame (unchanged)
            "disp_timestamp": 0.0,
            "z_m": None,                # estimated distance in meters
        },
        "oak1": {
            "frame": None,
            "target_center_x": None,
            "box": None,
            "score": None,
            "timestamp": 0.0,

            "disp": None,
            "disp_timestamp": 0.0,
            "z_m": None,
        },
    }

    async def camera_worker(sub, cam_name: str, stream_type: str):
        """
        stream_type: "left" or "disparity"
        """
        async for event, msg in oak_client.subscribe(sub, decode=True):
            now = time.time()

            # --- DISPARITY stream: store it and return ---
            if stream_type == "disparity":
                disp = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if disp is None:
                    continue
                states[cam_name]["disp"] = disp
                states[cam_name]["disp_timestamp"] = now
                continue

            # --- LEFT stream: run detection ---
            frame = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_height, frame_width = frame.shape[:2]

            # Downscale for YOLO speed
            if FRAME_SCALING != 1.0:
                yolo_frame = cv2.resize(frame, None, fx=FRAME_SCALING, fy=FRAME_SCALING)
            else:
                yolo_frame = frame

            async with inference_lock:
                results = await asyncio.to_thread(
                    model.predict,
                    source=yolo_frame,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU,
                    classes=[0],  # person
                    verbose=False,
                )

            best_box = None
            best_score = None
            center_x = None
            z_m = None

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()

                    # Choose largest area as "closest" within a camera
                    areas = (boxes_xyxy[:, X2] - boxes_xyxy[:, X1]) * (boxes_xyxy[:, Y2] - boxes_xyxy[:, Y1])
                    largest_area_id = int(np.argmax(areas))

                    x1, y1, x2, y2 = boxes_xyxy[largest_area_id].tolist()
                    best_score = float(scores[largest_area_id])

                    # Rescale bbox to full-res
                    if FRAME_SCALING != 1.0:
                        x1 /= FRAME_SCALING
                        y1 /= FRAME_SCALING
                        x2 /= FRAME_SCALING
                        y2 /= FRAME_SCALING

                    # Clamp bbox
                    x1 = int(clamp(x1, 0, frame_width - 1))
                    y1 = int(clamp(y1, 0, frame_height - 1))
                    x2 = int(clamp(x2, 0, frame_width - 1))
                    y2 = int(clamp(y2, 0, frame_height - 1))

                    best_box = (x1, y1, x2, y2)
                    center_x = (x1 + x2) // 2

                    # Compute depth from latest disparity (if available)
                    disp = states[cam_name].get("disp", None)
                    if disp is not None:
                        z_m = median_depth_from_disparity(
                            disp=disp,
                            box=best_box,
                            f_px=F_PX,
                            baseline_m=baseline_m,
                            disp_scale=disp_scale,
                        )

                    # Draw bbox + center
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, (y1 + y2) // 2), 6, (0, 0, 255), -1)

                    z_text = "z=??" if z_m is None else f"z={z_m:.2f}m"
                    cv2.putText(
                        frame,
                        f"{cam_name} conf={best_score:.2f} {z_text}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            # EMA smoothing for center x
            target_center_x = states[cam_name]["target_center_x"]
            if center_x is not None:
                if target_center_x is None:
                    target_center_x = float(center_x)
                else:
                    target_center_x = (EMA_ALPHA * target_center_x) + ((1.0 - EMA_ALPHA) * float(center_x))

            # Update state (LEFT frame is the main "frame" for display/control)
            states[cam_name]["frame"] = frame
            states[cam_name]["target_center_x"] = target_center_x
            states[cam_name]["box"] = best_box
            states[cam_name]["score"] = best_score
            states[cam_name]["z_m"] = z_m
            states[cam_name]["timestamp"] = now

    async def control_loop():
        period = 1.0 / SEND_HZ
        last_sent = 0.0

        last_valid_twist = Twist2d()
        last_valid_twist.linear_velocity_x = 0.0
        last_valid_twist.angular_velocity = 0.0
        last_detection_time = None
        stop_factor = 0.0

        cv2.namedWindow("oak0", cv2.WINDOW_NORMAL)
        cv2.namedWindow("oak1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("ACTIVE_TARGET", cv2.WINDOW_NORMAL)

        while True:
            now = time.time()

            # Choose closest target across cameras using smallest z_m
            best_cam = None
            best_z = 1e9

            for cam_name, cam_state in states.items():
                if cam_state["frame"] is not None:
                    cv2.imshow(cam_name, cam_state["frame"])

                age = now - cam_state["timestamp"]
                if age > LOST_TIMEOUT:
                    continue

                if cam_state["box"] is None or cam_state["target_center_x"] is None:
                    continue

                z_m = cam_state.get("z_m", None)
                if z_m is None:
                    continue

                if z_m < best_z:
                    best_z = z_m
                    best_cam = cam_name

            twist = Twist2d()
            twist.linear_velocity_x = 0.0
            twist.angular_velocity = 0.0

            if best_cam is not None:
                cam_state = states[best_cam]
                frame = cam_state["frame"]
                target_center_x = cam_state["target_center_x"]
                box = cam_state["box"]
                z_m = cam_state["z_m"]
                score = cam_state["score"]

                height, width = frame.shape[:2]
                center = width // 2
                horizontal_deadzone = int(width * ANGULAR_DEADZONE)

                # DEPTH-based distance control
                z_error = TARGET_Z_M - float(z_m)  # + => too far => forward; - => too close => reverse
                if abs(z_error) <= Z_DEADZONE_M:
                    linear_command = 0.0
                else:
                    linear_command = KP_LINEAR_Z * z_error
                    linear_command = clamp(linear_command, -MAX_REVERSE, MAX_FORWARD)

                # Heading control (same as before)
                angular_error = float(target_center_x - center)
                if FLIP_STEER:
                    angular_error = -angular_error

                if abs(angular_error) <= horizontal_deadzone:
                    angular_command = 0.0
                else:
                    angular_command = KP_ANGULAR * (angular_error / max(1, center))
                    angular_command = clamp(angular_command, -MAX_ANGULAR, MAX_ANGULAR)

                twist.linear_velocity_x = float(linear_command)
                # twist.angular_velocity = float(angular_command)  # uncomment when ready

                last_valid_twist.linear_velocity_x = float(linear_command)
                # last_valid_twist.angular_velocity = float(angular_command)
                last_detection_time = now

                # Display active feed
                active = frame.copy()
                cv2.line(active, (center, 0), (center, height), (255, 255, 0), 2)
                cv2.putText(active, f"ACTIVE: {best_cam}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                cv2.putText(active, f"z={z_m:.2f}m target={TARGET_Z_M:.2f} lin={linear_command:.2f} ang={angular_command:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(active, f"conf={0.0 if score is None else score:.2f} flip_steer={FLIP_STEER}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(active, f"baseline={baseline_m:.3f}m disp_scale={disp_scale:.1f}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("ACTIVE_TARGET", active)

            else:
                # target lost => soft stop
                if last_detection_time is not None:
                    time_since_lost = now - last_detection_time
                else:
                    time_since_lost = 99999.0

                if time_since_lost < LOST_TIMEOUT:
                    stop_factor = 1.0 - (time_since_lost / LOST_TIMEOUT)
                    twist.linear_velocity_x = last_valid_twist.linear_velocity_x * stop_factor
                    # twist.angular_velocity = last_valid_twist.angular_velocity * stop_factor
                else:
                    twist.linear_velocity_x = 0.0
                    twist.angular_velocity = 0.0
                    last_valid_twist = Twist2d()

                blank = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "NO TARGET (stopping)", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("ACTIVE_TARGET", blank)

            # Quit
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                await canbus_client.request_reply("/twist", Twist2d())
                return

            if now - last_sent >= period:
                await canbus_client.request_reply("/twist", twist)
                last_sent = now

            await asyncio.sleep(0.001)

    # Start workers for each subscription
    cam_tasks = []
    for sub in oak_config.subscriptions:
        q = sub.uri.query or ""
        path = sub.uri.path

        if "service_name=oak0" in q:
            cam = "oak0"
        elif "service_name=oak1" in q:
            cam = "oak1"
        else:
            continue

        if path == "/left":
            cam_tasks.append(asyncio.create_task(camera_worker(sub, cam, "left")))
        elif path == "/disparity":
            cam_tasks.append(asyncio.create_task(camera_worker(sub, cam, "disparity")))

    try:
        await control_loop()
    finally:
        for t in cam_tasks:
            t.cancel()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", type=str, required=True)

    # Calibration / depth tuning
    ap.add_argument("--baseline-m", type=float, default=0.075, help="Stereo baseline in meters (default 0.075)")
    ap.add_argument("--disp-scale", type=float, default=16.0, help="Disparity scale factor (default 16.0)")
    ap.add_argument("--target-z", type=float, default=1.5, help="Follow distance in meters (default 1.5)")

    args = ap.parse_args()

    asyncio.run(
        follow(
            ip=args.ip,
            baseline_m=args.baseline_m,
            disp_scale=args.disp_scale,
            target_z_m=args.target_z,
        )
    )