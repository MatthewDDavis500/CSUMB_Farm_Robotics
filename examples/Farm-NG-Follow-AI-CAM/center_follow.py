import cv2
import socket
import depthai as dai
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector

# TCP Connection Setup
ROBOT_IP = "100.87.161.11"
PORT = 9999
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ROBOT_IP, PORT))
print("[Follower] Connected to robot.")

# Initialize pose and hand detectors
pose_detector = PoseDetector()
hand_detector = HandDetector(detectionCon=0.8, maxHands=1)

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
frame_height = 720
frame_center = frame_width // 2
center_tolerance = frame_width // 10  # Tolerance for centering

# Thresholds for distance zones
LOWER_HEIGHT = 600
UPPER_HEIGHT = 800
TOO_CLOSE_WIDTH_RATIO = 0.9
TOO_CLOSE_HEIGHT_RATIO = 0.9

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_PURPLE = (255, 0, 255)
COLOR_RED = (0, 0, 255)

# State to track permanent stop after fist detection
stopped = False

# Run on device
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()

            # Pose detection
            img = pose_detector.findPose(frame)
            lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=True)

            # Hand detection and fist check (only if not stopped)
            if not stopped:
                hands, img = hand_detector.findHands(img, draw=True)
                if hands:
                    hand = hands[0]
                    fingers = hand_detector.fingersUp(hand)
                    if sum(fingers) == 0:
                        stopped = True
                        print("[Follower] Fist detected - stopping permanently.")
            else:
                # Still draw hands if stopped
                hands, img = hand_detector.findHands(img, draw=True)

            if stopped:
                command = 'x'
                box_color = None
            else:
                if bboxInfo is not None and 'bbox' in bboxInfo:
                    x, y, w, h = bboxInfo['bbox']
                    cx = x + w // 2
                    offset = cx - frame_center

                    # Too close → back away + adjust direction
                    if w >= TOO_CLOSE_WIDTH_RATIO * frame_width or h >= TOO_CLOSE_HEIGHT_RATIO * frame_height:
                        if abs(offset) < center_tolerance:
                            command = 's'  # Back straight
                        elif offset < 0:
                            command = 'a'  # Turn left while backing
                        else:
                            command = 'd'  # Turn right while backing
                        box_color = COLOR_RED

                    # Middle zone → stop but rotate to face
                    elif LOWER_HEIGHT <= h <= UPPER_HEIGHT:
                        if abs(offset) < center_tolerance:
                            command = 'x'  # Stay still
                        elif offset < 0:
                            command = 'a'  # Turn left
                        else:
                            command = 'd'  # Turn right
                        box_color = COLOR_PURPLE

                    # Far → follow logic (same as before)
                    elif h < LOWER_HEIGHT:
                        if abs(offset) < center_tolerance:
                            command = 'w'  # Move forward
                        elif offset < 0:
                            command = 'a'  # Turn left
                        else:
                            command = 'd'  # Turn right
                        box_color = COLOR_GREEN

                    else:
                        command = 'x'
                        box_color = COLOR_PURPLE

                    # Draw box + center dot
                    cv2.rectangle(img, (x, y), (x + w, y + h), box_color, thickness=3)
                    cv2.circle(img, (cx, y + h // 2), 6, (0, 0, 255), cv2.FILLED)

                else:
                    command = 'x'  # Stop if no detection
                    box_color = None

            # Send command
            try:
                client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")

            # Show frame
            cv2.imshow("Follower View", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('x'.encode())
                break

    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")