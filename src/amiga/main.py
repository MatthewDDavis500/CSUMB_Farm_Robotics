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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

### Config File Paths ###
OAK_CONFIG = str(ROOT_DIR / 'config' / 'oak_config.json')
CANBUS_CONFIG = str(ROOT_DIR / 'config' / 'canbus_config.json')

### Model Parameters ###
MODEL_NAME = str(ROOT_DIR / 'data' / 'models' / 'yolov8n.pt')    # YOLO model to load
CONFIDENCE_THRESHOLD = 0.65  # (0-1 Scale) How confident does the model need to be to consider an object a person?
IOU = 0.5                    # (0-1 Scale) Intersection over union. How much do two bounding boxes need to overlap for the model to consider them the same object and only take the one with the heighest confindence?
FRAME_SCALING = 0.6          # (Proportional Multiplier) Ratio by which to scale the camera frames before passing to model. Smaller numbers means smaller image size, which means faster computation but less accuracy
EMA_ALPHA = 0.85             # (0-1 Scale) Smoothness of reaction. Higher values means more memory (previous frames) is considered, so smoother but slower reactions. Lower values mean more new frames are considered, resulting in faster but possibly chattery responses

### Detection Box Vertex Indicies ###
X1 = 0
Y1 = 1
X2 = 2
Y2 = 3

### Distance Control Parameters ###
TARGET_HEIGHT = 0.8          # (Fraction) What fraction of the frame vertically should the detected person be ideally filling? Higher values means closer following
HEIGHT_DEADZONE = 0.05       # (Fraction) Deadzone for distance control. Having a detected person filling this much more or less than the TARGET_HEIGHT will be acceptable
KP_LINEAR = 0.8              # (Gain) Proportion for how fast the robot will move forward or backward based on the current error
MAX_FORWARD = 0.25           # (Meters/Second) Maximum forward velocity
MAX_REVERSE = 0.18           # (Meters/Second) Maximum reverse velocity

### Heading Control Parameters ###
FLIP_STEER = True            # (Boolean) If True, inverts steering
KP_ANGULAR = 1.2             # (Gain) Proportion for how fast the robot will turn based on the current error
MAX_ANGULAR = 0.5            # (Radians/Second) Maximum angular velocity
ANGULAR_DEADZONE = 0.1       # (Fraction) Deadzone for turning. Having a detected person within this much of the center will be acceptable

### Safety/Performance ###
LOST_TIMEOUT = 0.8           # (Seconds) How long the robot will continue movement before stopping completely if it doesn't detect anyone
SEND_HZ = 20.0               # (Hertz: X/Second) How many times per second twist commands will be sent to the motors


def clamp(value: float, min_val: float, max_val: float) -> float:
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
    temp = min(max_val, value)  # Constrain using the max value
    return float(max(min_val, temp)) # Constrain using the min value

async def follow():
    '''
        Asynchronous function for following a person with the robot.
        
        This function uses two different functions to operate:
            camera_worker(sub, cam_name) - Use the YOLO detection algorithm to detect people in camera frames.
            control_loop() - Use states updated by camera workers to send twist commands to the motors.
            
        Results:
            Robot moves to follow the closest person in the camera frames.
    '''
    # Import configuration from JSON config files
    canbus_config: EventServiceConfig = proto_from_json_file(CANBUS_CONFIG, EventServiceConfig())
    oak_config: EventServiceConfig = proto_from_json_file(OAK_CONFIG, EventServiceConfig())

    # Create clients for the canbus and oak cameras
    canbus_client = EventClient(canbus_config)
    oak_client = EventClient(oak_config)

    # Load the YOLO model
    model = YOLO(MODEL_NAME)

    # Per-camera latest state
    states = {
        'oak0': {
            "frame": None,            # Image from the Oak camera to be altered and displayed by OpenCV
            "target_center_x": None,  # Target person's horizontal center, smoothed to avoid motor jittering
            "box": None,              # Bounding box dimentions in full-res: (x1, y1, x2, y2)
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
        '''
        Asynchronous function for detecting people in the frame of a camera.

        Used here to determine how close a person is to the robot if there is a person in frame of one of the cameras.
        This data will be used to decide how much forward/backward the robot should move, if at all.
        
        Args:
            sub: The subscription of the camera to detect people in.
            cam_name: The name of the camera (ex. "oak0").
        
        Results:
            Updates the state entry corresponding to the camera name in the "states" dictionary
        '''
        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)

        # Asynchronously monitor the oak camera for people, using the YOLO model
        async for event, msg in oak_client.subscribe(sub, decode=True):
            # Decode the frame into a readable format, immediately skipping if the frame doesn't exist
            frame = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            
            frame_height, frame_width = frame.shape[:2]

            # Downscale the frame to speed up YOLO model (lower size = faster analysis)
            if FRAME_SCALING != 1.0:
                yolo_frame = cv2.resize(frame, None, fx=FRAME_SCALING, fy=FRAME_SCALING)
            else:
                yolo_frame = frame

            # Set up the YOLO model with our constant parameters
            results = model.predict(
                source=yolo_frame,          # Load our camera frame into the model
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU,
                classes=[0],                # Only detect people
                verbose=False,              # Do not output detection logs to terminal
            )

            best_box = None
            best_score = None
            center_x = None
            height_fraction = None

            # If the model returns results for any frames, look at the first (and only) frame
            if results and len(results) > 0:
                result = results[0]
                
                # If the model detected anything in the frame
                if result.boxes is not None and len(result.boxes) > 0:
                    # Bring the boxes (in x1,y1,x2,y2 format) and their corresponding confidence scores data from the GPU to the CPU in NumPy array format
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()

                    # Calculate the areas of all of the boxes, and find the index of the largest area (corresponding to the largest box)
                    areas = (boxes_xyxy[:, X2] - boxes_xyxy[:, X1]) * (boxes_xyxy[:, Y2] - boxes_xyxy[:, Y1])  # For all boxes, store the area (width (x2-x1) * height (y2-y1))
                    largest_area_id = int(np.argmax(areas))

                    # Use the index of the largest area to get the largest (and closest) detection result details
                    x1, y1, x2, y2 = boxes_xyxy[largest_area_id].tolist()
                    best_score = float(scores[largest_area_id])

                    # Rescale box vertices back to full-resolution if frame was previously downscaled
                    if FRAME_SCALING != 1.0:
                        x1 /= FRAME_SCALING; y1 /= FRAME_SCALING
                        x2 /= FRAME_SCALING; y2 /= FRAME_SCALING

                    # Ensure that the box remains completely within frame after rescaling
                    x1 = int(clamp(x1, 0, frame_width - 1))
                    y1 = int(clamp(y1, 0, frame_height - 1))
                    x2 = int(clamp(x2, 0, frame_width - 1))
                    y2 = int(clamp(y2, 0, frame_height - 1))

                    # Store the best bounding box and center x-value
                    best_box = (x1, y1, x2, y2)
                    center_x = (x1 + x2) // 2

                    # Find the fraction of the frame's height that the bounding box takes up (used to find how close it is)
                    box_height = max(1, (y2 - y1))  # Value will always be at least 1 so that fraction is never 0
                    height_fraction = box_height / float(frame_height)

                    # Draw bounding box on the frame for visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, (y1 + y2) // 2), 6, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"{cam_name} conf={best_score:.2f} h={height_fraction:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            # EMA smoothing for center x-value
            target_center_x = states[cam_name]["target_center_x"]
            if center_x is not None:
                if target_center_x is None:
                    target_center_x = float(center_x)
                else:
                    # Smooth out turning by choosing a point between the current point and the actual center of the bounding box to turn to
                    target_center_x = (EMA_ALPHA * target_center_x) + ((1.0 - EMA_ALPHA) * float(center_x))

            # Update state
            states[cam_name]["frame"] = frame
            states[cam_name]["target_center_x"] = target_center_x
            states[cam_name]["box"] = best_box
            states[cam_name]["height_fraction"] = height_fraction
            states[cam_name]["score"] = best_score
            states[cam_name]["timestamp"] = time.time()

    async def control_loop():
        '''
            Asynchronous function for controlling the robot's movement.

            Uses updating camera states to send twist commands to the robot's motors.
            
            Results:
                Sends twist commands to motors.
        '''
        period = 1.0 / SEND_HZ  # How many seconds between each motor twist message send
        last_sent = 0.0
        
        # These will be used to implement a slow stop when someone goes out of frame
        last_valid_twist = Twist2d()
        last_valid_twist.linear_velocity_x = 0.0
        last_valid_twist.angular_velocity = 0.0
        last_detection_time = None
        stop_factor = 0

        # Create a unique window for the camera feed with the closest detected person
        cv2.namedWindow("ACTIVE_TARGET", cv2.WINDOW_NORMAL)

        while True:
            now = time.time()

            # Choose the best/closest target across cameras
            best_cam = None
            best_height = -1.0  # Height fractions are always 0 to 1, so initializing to -1 ensures that any height is chosen over this initial value

            for cam_name, cam_state in states.items():
                # Show camera feed
                cv2.imshow(cam_name, cam_state['frame'])
                
                # Find how long ago the last state update was
                age = now - cam_state["timestamp"]
                
                # If the last time the state was updated was longer than the timeout constant, do nothing
                if age > LOST_TIMEOUT:
                    continue
                # If any critical information is missing from the state, do nothing
                if cam_state["box"] is None or cam_state["target_center_x"] is None or cam_state["height_fraction"] is None:
                    continue

                # Find the camera with the closest detected person (tallest bounding box)
                if cam_state["height_fraction"] > best_height:
                    best_height = cam_state["height_fraction"]
                    best_cam = cam_name

            # Create a twist command initialized to no movement
            twist = Twist2d()
            twist.linear_velocity_x = 0.0
            twist.angular_velocity = 0.0

            # Send a twist command to the robot motors based on the camera with the closest detected person. If there is none, stop.
            if best_cam is not None:
                # Get details from the camera state
                cam_state = states[best_cam]
                frame = cam_state["frame"]
                target_center_x = cam_state["target_center_x"]
                box = cam_state["box"]
                height_fraction = cam_state["height_fraction"]
                score = cam_state["score"]

                # Get details about frame dimensions
                height, width = frame.shape[:2]
                center = width // 2
                horizontal_deadzone = int(width * ANGULAR_DEADZONE)

                # Distance control based on height of bounding box
                # A positive error means the bounding box is shorter (and thus farther) than the target, so robot needs to move forward
                # A negative error means the bounding box is taller (and thus closer) than the target, so robot needs to move backward
                height_error = TARGET_HEIGHT - float(height_fraction)

                # Linear movement
                if abs(height_error) <= HEIGHT_DEADZONE:
                    # If error is within the height deadzone, no linear adjustment is needed
                    linear_command = 0.0
                else:
                    # Compute a linear velocity based on the current error and clamp it between velocity limits
                    linear_command = KP_LINEAR * height_error
                    linear_command = clamp(linear_command, -MAX_REVERSE, MAX_FORWARD)

                # Heading control based on bounding box's distance from the center of the frame
                # Positive/negative error will determine if robot needs to turn left/right to center on the bounding box
                angular_error = float(target_center_x - center)
                
                # Angular movement
                if FLIP_STEER:
                    angular_error = -angular_error

                if abs(angular_error) <= horizontal_deadzone:
                    # If error is within the horizontal deadzone, no angular adjustment is needed
                    angular_command = 0.0
                else:
                    # Compute an angular velocity based on the current error and clamp it between velocity limits
                    angular_command = KP_ANGULAR * (angular_error / center)
                    angular_command = clamp(angular_command, -MAX_ANGULAR, MAX_ANGULAR)

                # Store the linear and angluar velocities in the twist command
                twist.linear_velocity_x = float(linear_command)
                twist.angular_velocity = float(angular_command)
                
                # Update last valid twist
                last_valid_twist.linear_velocity_x = float(linear_command)
                last_valid_twist.angular_velocity = float(angular_command)
                last_detection_time = now

                # Display the active camera feed and twist details
                active = frame.copy()
                cv2.line(active, (center, 0), (center, height), (255, 255, 0), 2)
                cv2.putText(active, f"ACTIVE: {best_cam}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(active, f"height_fraction={height_fraction:.2f} target={TARGET_HEIGHT:.2f} lin={linear_command:.2f} ang={angular_command:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(active, f"conf={0.0 if score is None else score:.2f} flip_steer={FLIP_STEER}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("ACTIVE_TARGET", active)
            else:
                # Calculate exactly how long the target has been lost
                if(last_detection_time is not None):
                    time_since_lost = now - last_detection_time
                else:
                    time_since_lost = 99999.0
                # Slow stop when target lost
                if(time_since_lost < LOST_TIMEOUT):
                    # If still in timeout duration, gradually decrease speed based on how close to the end of timeout duration
                    stop_factor = 1.0 - (time_since_lost / LOST_TIMEOUT)
                    twist.linear_velocity_x = last_valid_twist.linear_velocity_x * stop_factor
                    twist.angular_velocity = last_valid_twist.angular_velocity * stop_factor
                else:
                    # If timeout duration reached, stop robot completely and reset last valid twist
                    twist.linear_velocity_x = 0.0
                    twist.angular_velocity = 0.0
                    last_valid_twist = Twist2d()
            
                # No target, so display empty target window
                blank = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "NO TARGET (stopping)", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("ACTIVE_TARGET", blank)

            # Quit if q pressed in any window
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                await canbus_client.request_reply("/twist", Twist2d())
                return

            # Publish at fixed rate by waiting until the specified amount of seconds has passed since last publish
            if now - last_sent >= period:
                await canbus_client.request_reply("/twist", twist)  # Send the twist command to the robot
                last_sent = now  # Update time last sent

            await asyncio.sleep(0.001)

    # Run camera workers and control loop
    cam_tasks = []
    for sub in oak_config.subscriptions:
        # Determine camera name
        if 'oak0' in sub['uri']['query']:
            cam_tasks.append(asyncio.create_task(camera_worker(sub, 'oak0')))
        elif 'oak1' in sub['uri']['query']:
            cam_tasks.append(asyncio.create_task(camera_worker(sub, 'oak1')))
        
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