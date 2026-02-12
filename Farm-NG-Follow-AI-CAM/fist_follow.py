#This version of the follow program stops using fist detectioon
#When a fist is detected it will send a stop command
#When it doesnt detect a fist it will continue following again

import cv2
import socket
import depthai as dai
import mediapipe as mp
from cvzone.PoseModule import PoseDetector

# TCP Connection Setup
ROBOT_IP = "100.87.161.11"
PORT = 9999
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ROBOT_IP, PORT))
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

# Create pipeline and camera
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(1280, 720)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Frame and movement setup
frame_width = 1280
frame_center = frame_width // 2
center_tolerance = frame_width // 10

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()

            # Pose detection
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
                client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")
                break

            # Show annotated frame
            cv2.imshow("Follower View", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('x'.encode())
                break

    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")