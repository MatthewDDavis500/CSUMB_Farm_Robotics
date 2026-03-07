import cv2
import socket
import depthai as dai
from cvzone.PoseModule import PoseDetector

# TCP Connection Setup
ROBOT_IP = "100.87.161.11"
PORT = 9999
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ROBOT_IP, PORT))
print("[Follower] Connected to robot.")

# Initialize pose detector
pose_detector = PoseDetector()

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

# Thresholds for distance zones (adjust these for your setup)
LOWER_HEIGHT = 700   # below this -> move forward (green)
UPPER_HEIGHT = 850   # between LOWER and UPPER -> stop (purple)
TOO_CLOSE_WIDTH_RATIO = 0.9   # width or height > 90% of frame -> move backward (red)
TOO_CLOSE_HEIGHT_RATIO = 0.9

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_PURPLE = (255, 0, 255)  # violet/purple
COLOR_RED = (0, 0, 255)

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

            if bboxInfo is not None and 'bbox' in bboxInfo:
                x, y, w, h = bboxInfo['bbox']

                cx = x + w // 2
                offset = cx - frame_center

                # Decide movement command and bbox color
                if w >= TOO_CLOSE_WIDTH_RATIO * frame_width or h >= TOO_CLOSE_HEIGHT_RATIO * frame_height:
                    command = 's'  # move backward
                    box_color = COLOR_RED
                elif LOWER_HEIGHT <= h <= UPPER_HEIGHT:
                    command = 'x'  # stop
                    box_color = COLOR_PURPLE
                elif h < LOWER_HEIGHT:
                    # move forward or turn
                    if abs(offset) < center_tolerance:
                        command = 'w'
                    elif offset < 0:
                        command = 'a'
                    else:
                        command = 'd'
                    box_color = COLOR_GREEN
                else:
                    # fallback, stop
                    command = 'x'
                    box_color = COLOR_PURPLE

                # Draw bounding box and center dot with dynamic color
                cv2.rectangle(img, (x, y), (x + w, y + h), box_color, thickness=3)
                cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)

            else:
                command = 'x'  # stop if no person detected
                # No bounding box drawn if no person detected

            try:
                client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")

            # Show the full annotated frame
            cv2.imshow("Follower View", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('x'.encode())
                break

    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")