#This version of the follow program stops using fist detectioon
#When a fist is detected it will send a stop command
#When it doesnt detect a fist it will continue following again

import asyncio
import cv2
import socket
import mediapipe as mp
from cvzone.PoseModule import PoseDetector

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.canbus.canbus_pb2 import Twist2d
import numpy as np

# TCP Connection Setup
ROBOT_IP = "100.75.188.55"
# PORT = 50011
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect((ROBOT_IP, PORT))
print("[Follower] Connected to robot.")

# Initialize pose detector
pose_detector = PoseDetector()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Helper: check if hand is a fist
def is_fist(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip(tips, pips))

# Frame and movement setup
frame_width = 1280
frame_center = frame_width // 2
center_tolerance = frame_width // 10

# 1. Setup the Camera Client and CANBUS client
cam_config = EventServiceConfig(name="oak0", host=ROBOT_IP, port=50011)
camera_client = EventClient(cam_config)

motor_client = EventClient(EventServiceConfig(name="canbus", host=ROBOT_IP, port=50051))

async def test_camera():
    ROBOT_IP = "100.75.188.55"
    PORT = 50010 # Standard OAK port
    
    config = EventServiceConfig(name="test_cam", host=ROBOT_IP, port=PORT)
    client = EventClient(config)
    
    print(f"Checking camera at {ROBOT_IP}:{PORT}...")
    
    try:
        # We use a timeout so it doesn't hang forever
        async for event, payload in asyncio.wait_for(client.subscribe(config), timeout=5.0):
            print(f"SUCCESS! Received frame: {len(payload)} bytes")
            break # We only need one frame to confirm
    except asyncio.TimeoutError:
        print("FAILURE: Connection timed out. The Amiga is there, but no video is flowing.")
    except Exception as e:
        print(f"ERROR: {e}")

async def main_loop():
    try:
        # 2. Start the camera stream
        async for event, payload in camera_client.subscribe(cam_config):
            # Convert the Amiga stream to an OpenCV frame
            # (The Amiga sends raw bytes that need decoding)
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            img = pose_detector.findPose(frame)
            lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=True)

            # Hand detection for fist
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            fist_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_fist(hand_landmarks.landmark):
                        fist_detected = True
                    # Draw hand bounding box for visualization
                    h, w, _ = frame.shape
                    x_vals = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_vals = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    x1, y1 = min(x_vals), min(y_vals)
                    x2, y2 = max(x_vals), max(y_vals)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            if fist_detected:
                command = 'x'  # STOP completely
            elif bboxInfo is not None and 'bbox' in bboxInfo:
                x, y, w, h = bboxInfo['bbox']
                cx = x + w // 2
                offset = cx - frame_center

                # Draw bounding box and center dot
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)

                # Movement based on center offset
                if abs(offset) < center_tolerance:
                    command = 'w'  # forward
                elif offset < 0:
                    command = 'a'  # turn left
                else:
                    command = 'd'  # turn right
            else:
                command = 'x'  # no person detected, stop

            try:
                # client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")
                break

            # Show annotated frame
            cv2.imshow("Follower View", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # client_socket.sendall('x'.encode())
                break
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main_loop())