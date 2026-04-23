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
OAK_CONFIG = str(ROOT_DIR / 'config' / 'oak_config.json')
CANBUS_CONFIG = str(ROOT_DIR / 'config' / 'canbus_config.json')

### Model Parameters ###
MODEL_NAME = str(ROOT_DIR / 'data' / 'models' / 'yolov8n.pt')    # YOLO model to load
CONFIDENCE_THRESHOLD = 0.65  # (0-1 Scale) How confident does the model need to be to consider an object a person?
IOU = 0.5                    # (0-1 Scale) Intersection over union. How much do two bounding boxes need to overlap for the model to consider them the same object and only take the one with the heighest confindence?
FRAME_SCALING = 1.0          # (Proportional Multiplier) Ratio by which to scale the camera frames before passing to model. Smaller numbers means smaller image size, which means faster computation but less accuracy
EMA_ALPHA = 0.85             # (0-1 Scale) Smoothness of reaction. Higher values means more memory (previous frames) is considered, so smoother but slower reactions. Lower values mean more new frames are considered, resulting in faster but possibly chattery responses

### Detection Box Vertex Indices ###
X1, Y1, X2, Y2 = 0, 1, 2, 3

### Distance Control Parameters ###
TARGET_DEPTH = 1.5           # (Meters) Target distance from which to follow a person 
DEPTH_DEADZONE = 0.15        # (Meters) Deadzone for following. Having a detected person within this much of the target depth will be acceptable
CAMERA_BASELINE = 0.075      # (Meters) Distance between the centers of both camera lenses used by OAK-Ds for depth. Used for triangulation calculations
DISPARITY_SCALE = 0.33         # 
KP_LINEAR = 0.7              # (Gain) Proportion for how fast the robot will move forward or backward based on the current error
MAX_FORWARD = 0.3            # (Meters/Second) Maximum forward velocity
MAX_REVERSE = 0.3            # (Meters/Second) Maximum reverse velocity

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
    return float(max(min_val, min(max_val, value)))

def visualize_depth(raw_disparity):
    '''Colorizes disparity for debugging.'''
    if raw_disparity is None: 
        return None
    
    # Normalize disparity to 0-255 (grayscale)
    disparity_visualization = (raw_disparity / DISPARITY_SCALE).astype(np.uint8)
    return cv2.applyColorMap(disparity_visualization, cv2.COLORMAP_JET)

def median_depth_from_disparity(disparity, box, focal_length):
    '''
        Finds the median depth of an area of a box in a disparity frame.
        
        Used here to determine how far a detected person is from the camera for accurate following from a distance.
        
        Args:
            disparity: A frame from the OAK-D's disparity stream.
            box: The bounding box of the detected person.
            focal_length: Feature of the camera that determines FOV, important for translating meters into pixels.
            
        Returns:
            The depth of the detected person in the frame (how far they are) in meters.
    '''
    # If any critical inputs are missing, return None
    if disparity is None or box is None or focal_length <= 0: 
        return None
    
    # Get box vertices, width, and height
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    
    # Get disparity frame's dimensions
    HEIGHT, WIDTH = disparity.shape[:2]

    # Calculate Region Of Interest (ROI) (Center 30% of the bounding box) to avoid getting depth of stuff behind the detected person
    rx1 = int(clamp(x1 + 0.3*box_width, 0, WIDTH-1))
    rx2 = int(clamp(x2 - 0.3*box_width, 0, WIDTH))
    ry1 = int(clamp(y1 + 0.3*box_height, 0, HEIGHT-1))
    ry2 = int(clamp(y2 - 0.1*box_height, 0, HEIGHT))

    # If vertices of ROI box are invalid, return None
    if rx2 <= rx1 or ry2 <= ry1: 
        return None

    region_of_interest = disparity[ry1:ry2, rx1:rx2].astype(np.float32)
    real_disparity = region_of_interest / float(DISPARITY_SCALE)  # Use the disparity scale to get the actual disparity
    real_disparity = real_disparity[np.isfinite(real_disparity)]  # Only get disparity that isn't infinite or NaN
    real_disparity = real_disparity[real_disparity > 0.5]  # Filter noise
    
    # Require minimum valid pixels
    if real_disparity.size < 50: 
        return None 

    # Find the median disparity
    real_disparity_median = np.median(real_disparity)
    if real_disparity_median <= 0: 
        return None
    
    # Calculate the depth of the detected person
    depth = (focal_length * CAMERA_BASELINE) / real_disparity_median
    
    # Return None if depth isn't in the range of valid values
    if 0.3 < depth < 15.0:
        return depth  
    else:
        return None

async def follow(ip: str):
    '''
        Asynchronous function for following a person with the robot.
        
        This function uses two different functions to operate:
            camera_worker(sub, cam_name) - Use the YOLO detection algorithm to detect people in camera frames.
            control_loop() - Use states updated by camera workers to send twist commands to the motors.
            
        Results:
            Robot moves to follow the closest person in the camera frames.
    '''
    # Import configuration from JSON config files
    canbus_config = proto_from_json_file(CANBUS_CONFIG, EventServiceConfig())
    oak_config = proto_from_json_file(OAK_CONFIG, EventServiceConfig())
    
    # Load user-provided IP into config
    canbus_config.host = ip
    oak_config.host = ip 

    # Create clients for the canbus and oak cameras
    canbus_client = EventClient(canbus_config)
    oak_client = EventClient(oak_config)

    # Calibration Fetching
    try:
        print('Requesting calibration details...')
        calibration = await oak_client.request_reply('/calibration', Empty(), decode=True)  # Request OAK calibration details
        
        # The focal length is the distance between the camera lens and the light sensor. A short focal length would mean a wide FOV. Used in depth calculations
        CAMERA_FOCAL_LENGTH = float(calibration.camera_data[0].intrinsic_matrix[0])
        print(f'[CALIBRATION] Successfully loaded focal length = {CAMERA_FOCAL_LENGTH}')
    except Exception as e:
        CAMERA_FOCAL_LENGTH = 800.0 # Default for OAK-D at 720p
        print(f'[WARN] Calibration failed: {e}. Using default focal length: {CAMERA_FOCAL_LENGTH}')

    # Load the YOLO model
    model = YOLO(MODEL_NAME)
    
    # Lock ensures that only one camera worker uses the model at a time
    inference_lock = asyncio.Lock()

    # Per-camera latest state
    states = {
        'oak0': {
            'frame': None,            # Image from the Oak camera to be altered and displayed by OpenCV
            'target_center_x': None,  # Target person's horizontal center, smoothed to avoid motor jittering
            'box': None,              # Bounding box dimentions in full-res: (x1, y1, x2, y2)
            'score': None,            # Model confidence in person detection
            'timestamp': 0.0,         # Timestamp of current state
            'last_detection': 0.0,    # Timestamp of the last frame with a detected person
            'disparity': None,        # Disparity frame from OAK camera
            'depth': None,            # Calculated depth from detected person
        },
        'oak1': {
            'frame': None,
            'target_center_x': None,
            'box': None,
            'score': None,
            'timestamp': 0.0,
            'last_detection': 0.0,
            'disparity': None,
            'depth': None,
        }
    }

    async def camera_worker(sub, cam_name: str, stream_type: str):
        '''
        Asynchronous function for detecting people in the frame of a camera.

        Used here to determine how close a person is to the robot if there is a person in frame of one of the cameras.
        This data will be used to decide how much forward/backward the robot should move, if at all.
        
        Args:
            sub: The subscription of the camera to detect people in.
            cam_name: The name of the camera (ex. 'oak0').
        
        Results:
            Updates the state entry corresponding to the camera name in the "states" dictionary
        '''
        # Asynchronously monitor the oak camera for detected people, using the YOLO model
        async for event, msg in oak_client.subscribe(sub, decode=True):
            # If there are more messages waiting, skip this one
            # if oak_client.get_queue_size(sub) > 1:
            #     print('skipping frame...')
            #     continue
            # print('processing this frame!')
            
            now = time.time()

            # If this camera stream is disparity, just use it to update the disparity value for that camera's state
            if stream_type == 'disparity':
                disparity = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if disparity is not None:
                    states[cam_name]['disparity'] = disparity
                continue

            # Decode the frame into a readable format, immediately skipping if the frame doesn't exist
            frame = cv2.imdecode(np.frombuffer(msg.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: 
                continue
            
            # Downscale the frame to speed up YOLO model (lower size = faster analysis)
            yolo_frame = cv2.resize(frame, None, fx=FRAME_SCALING, fy=FRAME_SCALING) if FRAME_SCALING != 1.0 else frame

            # Run the YOLO model with our constant parameters, using the asynch lock to ensure only this worker can use the model right now
            async with inference_lock:
                results = await asyncio.to_thread(
                    model.predict,              # Load our camera frame into the model
                    source=yolo_frame, 
                    conf=CONFIDENCE_THRESHOLD, 
                    iou=IOU, 
                    classes=[0],                # Only detect people
                    verbose=False,              # Do not output detection logs to terminal
                )

            # If the model returns results for any frames
            if results and len(results[0].boxes) > 0:
                result = results[0]
                # Calculate the areas of all of the boxes, and find the index of the largest area (corresponding to the largest box)
                areas = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]) * (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
                largest_area_id = int(np.argmax(areas.cpu().numpy()))
                
                # Use the index of the largest area to get the largest (and closest) detection result details
                box = result.boxes.xyxy[largest_area_id].cpu().numpy()
                
                # Rescale box vertices back to full-resolution if frame was previously downscaled
                if(FRAME_SCALING != 1.0):
                    box /= FRAME_SCALING
                    
                # Get the two corners of the bounding box
                x1, y1, x2, y2 = box.astype(int)
                
                # Update State
                states[cam_name]['box'] = (x1, y1, x2, y2)
                states[cam_name]['score'] = float(result.boxes.conf[largest_area_id])
                states[cam_name]['last_detection'] = now
                
                # Calculate depth of the person in the frame
                raw_z = median_depth_from_disparity(
                    states[cam_name]['disparity'], 
                    (x1,y1,x2,y2),
                    CAMERA_FOCAL_LENGTH
                )
                
                # If a depth was found, apply EMA smoothing
                if raw_z:
                    # Apply EMA smoothing to Z distance
                    prev_depth = states[cam_name]['depth']
                    if prev_depth is None:
                        states[cam_name]['depth'] = raw_z  
                    else:
                        states[cam_name]['depth'] = (EMA_ALPHA * prev_depth) + ((1-EMA_ALPHA) * raw_z)
                
                # EMA smoothing for center x-value
                center_x = (x1 + x2) // 2
                prev_center_x = states[cam_name]['target_center_x']
                
                # Smooth out turning by choosing a point between the current point and the actual center of the bounding box to turn to
                if prev_center_x is None:
                    states[cam_name]['target_center_x'] = float(center_x)  
                else:
                    states[cam_name]['target_center_x'] = (EMA_ALPHA * prev_center_x) + ((1-EMA_ALPHA) * center_x)

                # Draw bounding box on the frame for visualization
                cv2.rectangle(
                    frame, 
                    (x1, y1), 
                    (x2, y2), 
                    (0, 255, 0), 
                    2
                )
                
                # Draw the 30% ROI box used for depth
                box_width = x2 - x1
                box_height = y2 - y1
                cv2.rectangle(
                    frame, 
                    (x1 + int(0.3 * box_width), y1 + int(0.3 * box_height)), 
                    (x2 - int(0.3 * box_width), y2 - int(0.1 * box_height)), 
                    (255, 0, 0), 
                    1
                )

            # Update state with new frame
            states[cam_name]['frame'] = frame
            states[cam_name]['timestamp'] = now

    async def control_loop():
        '''
            Asynchronous function for controlling the robot's movement.

            Uses updating camera states to send twist commands to the robot's motors.
            
            Results:
                Sends twist commands to motors.
        '''
        period = 1.0 / SEND_HZ  # How many seconds between each motor twist message send
        
        # Used for slow stop when someone leaves the frame
        last_sent = 0.0
        last_valid_twist = Twist2d()
        last_detection_time = None
        
        # Create windows for camera feeds
        cv2.namedWindow('ACTIVE_TARGET', cv2.WINDOW_NORMAL)

        while True:
            now = time.time()
            
            # print(f'Last Oak0 detection: {states['oak0']['last_detection']}')
            # print(f'Last Oak1 detection: {states['oak1']['last_detection']}')
            # print(f'Last Oak0 dpeth: {states['oak0']['depth']}')
            # print(f'Last Oak1 depth: {states['oak1']['depth']}')
            # print('-----------------------------------------------------------------------------------------------')
            
            
            # Variables for storing best/closest target across cameras
            best_cam = None
            best_depth = 1e9  # Initialized to very far away so that any distance is considered better than the initial
            
            # Choose the best/closest target across cameras
            for cam, state in states.items():
                # Show camera feed
                if state['frame'] is not None: 
                    cv2.imshow(cam, state['frame'])
                if state['disparity'] is not None: 
                    cv2.imshow(f'disparity_{cam}', visualize_depth(state['disparity']))
                
                if (now - state['last_detection']) < LOST_TIMEOUT and state['depth']:
                    if state['depth'] < best_depth:
                        best_depth = state['depth']
                        best_cam = cam

            # Create a twist command initialized to no movement
            twist = Twist2d()
            
            # Send a twist command to the robot motors based on the camera with the closest detected person. If there is none, stop.
            if best_cam:
                state = states[best_cam]
                depth_error = state['depth'] - TARGET_DEPTH # Positive = too far, Negative = too close
                
                # Compute a linear velocity based on the current error
                if abs(depth_error) > DEPTH_DEADZONE:
                    linear_velocity = KP_LINEAR * depth_error  
                else:
                    linear_velocity = 0.0
                
                # Set the twist linear velocity, clamped between velocity limits
                twist.linear_velocity_x = -clamp(linear_velocity, -MAX_REVERSE, MAX_FORWARD)

                # Heading control based on bounding box's distance from the center of the frame
                angular_error = (state['target_center_x'] - (state['frame'].shape[1] // 2))
                if FLIP_STEER:
                    angular_error = -angular_error
                    
                # Compute an angular velocity based on the current error
                if abs(angular_error) > (state['frame'].shape[1] * ANGULAR_DEADZONE):  # error > (frame_width * deadzone)
                    angular_velocity = KP_ANGULAR * (angular_error / (state['frame'].shape[1] / 2))  
                else:
                    angular_velocity = 0.0
                    
                # Set the twist angular velocity, clamped between velocity limits
                # twist.angular_velocity = clamp(angular_velocity, -MAX_ANGULAR, MAX_ANGULAR)

                # Update last valid twist
                last_valid_twist = twist
                last_detection_time = now
                
                # Display the active camera feed and twist details
                active = state['frame'].copy()
                center = active.shape[1] // 2
                cv2.line(active, (center, 0), (center, active.shape[0]), (255, 255, 0), 2)
                cv2.putText(active, f"ACTIVE: {best_cam}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(active, f"depth={state['depth']:.2f} target={TARGET_DEPTH:.2f} lin={linear_velocity:.2f} ang={angular_velocity:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if state['score'] is None:
                    cv2.putText(active, f"conf=0.0 flip_steer={FLIP_STEER}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(active, f"conf={state['score']:.2f} flip_steer={FLIP_STEER}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("ACTIVE_TARGET", active)
            else:
                # Slow stop when not detecting a person
                if last_detection_time and (now - last_detection_time) < LOST_TIMEOUT:
                    # If still in timeout duration, gradually decrease speed based on how close to the end of timeout duration
                    factor = 1.0 - ((now - last_detection_time) / LOST_TIMEOUT)
                    twist.linear_velocity_x = last_valid_twist.linear_velocity_x * factor
                    # twist.angular_velocity = last_valid_twist.angular_velocity * factor
                else:
                    # If timeout duration reached, stop robot completely and reset last valid twist
                    last_valid_twist = Twist2d()
                    
                # Create a blank or dimmed frame to show we are searching
                searching_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(searching_frame, "LOST TARGET - SEARCHING...", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("ACTIVE_TARGET", searching_frame)

            # Publish at fixed rate by waiting until the specified amount of seconds has passed since last publish
            if now - last_sent >= period:
                await canbus_client.request_reply('/twist', twist) # Send the twist command to the robot
                last_sent = now

            # Quit if q pressed in any window
            if (cv2.waitKey(1) & 0xFF) == ord('q'): 
                break
            
            await asyncio.sleep(0.005)

    # Subscription Setup
    tasks = []
    for sub in oak_config.subscriptions:
        # Determine camera name
        if 'oak0' in sub.uri.query:
            name = 'oak0'  
        elif 'oak1' in sub.uri.query:
            name = 'oak1'
        else:
            continue
        
        # Determine stream type
        if '/left' in sub.uri.path:
            stream_type = 'left'
        elif '/disparity' in sub.uri.path:
            stream_type = 'disparity'
        else:
            continue
        
        # If the camera name and stream type were both identified, assign it to a camera worker
        tasks.append(asyncio.create_task(camera_worker(sub, name, stream_type)))

    try:
        await control_loop()
    finally:
        for t in tasks: t.cancel()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=True)
    args = parser.parse_args()
    asyncio.run(follow(args.ip))