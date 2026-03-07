#Hold fist to pause follow functions
#Lower fist to resume follow
#Will move backwards whenever a person is too close
#If the person’s width is ≥ 90% of the frame width
#Or if the person’s height is ≥ 90% of the frame height

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
hand_detector = HandDetector(detectionCon=0.8, maxHands=1)  # Adjust detectionCon if needed

# Create pipeline and camera
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(1280, 720)  # Higher resolution for better range
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Frame and movement setup
frame_width = 1280
frame_height = 720
frame_center = frame_width // 2
center_tolerance = frame_width // 10  # acceptable range to go straight

# Thresholds for distance zones (adjust for your space)
LOWER_HEIGHT = 500   # Below this: move forward
UPPER_HEIGHT = 1000   # Between these: sweet zone (stop)
TOO_CLOSE_WIDTH_RATIO = 0.9   # If person fills 90% of frame
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

            # Hand detection and fist check (only if not stopped), now drawing landmarks and bbox
            if not stopped:
                hands, img = hand_detector.findHands(img, draw=True)  # draw=True draws landmarks & connections
                
                if hands:
                    for hand in hands:
                        # Draw bounding box around hand
                        xH, yH, wH, hH = hand['bbox']
                        cv2.rectangle(img, (xH, yH), (xH + wH, yH + hH), (255, 255, 0), 2)  # Cyan box

                    hand = hands[0]
                    fingers = hand_detector.fingersUp(hand)
                    if sum(fingers) == 0:  # all fingers down = fist detected
                        stopped = True
                        print("[Follower] Fist detected - stopping permanently.")

            if stopped:
                command = 'x'  # Permanently stop
                box_color = None
            else:
                if bboxInfo is not None and 'bbox' in bboxInfo:
                    x, y, w, h = bboxInfo['bbox']
                    cx = x + w // 2
                    offset = cx - frame_center

                    # Movement logic and bounding box color
                    if w >= TOO_CLOSE_WIDTH_RATIO * frame_width or h >= TOO_CLOSE_HEIGHT_RATIO * frame_height:
                        command = 's'  # Move backward
                        box_color = COLOR_RED
                    elif LOWER_HEIGHT <= h <= UPPER_HEIGHT:
                        command = 'x'  # Stop
                        box_color = COLOR_PURPLE
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

                    # Draw full bounding box and center dot
                    cv2.rectangle(img, (x, y), (x + w, y + h), box_color, thickness=3)
                    cv2.circle(img, (cx, y + h // 2), 6, (0, 0, 255), cv2.FILLED)
                else:
                    command = 'x'  # Stop if no person
                    box_color = None

            # Send command
            try:
                client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")

            # Show frame with pose and hand landmarks + bounding boxes
            cv2.imshow("Follower View", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('x'.encode())
                break

    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")