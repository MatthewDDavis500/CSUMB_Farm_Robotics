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
frame_center = frame_width // 2
center_tolerance = frame_width // 10  # acceptable range to go straight

# Track permanent stop state
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

            # Only use bounding box height to decide stop
            if bboxInfo is not None and 'bbox' in bboxInfo:
                x, y, w, h = bboxInfo['bbox']

                # Stop if person is too close
                if h > 900:  # Adjust threshold based on your testing
                    command = 'x'
                else:
                    cx = x + w // 2
                    offset = cx - frame_center

                    # Draw bounding box and center dot
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)

                    if abs(offset) < center_tolerance:
                        command = 'w'
                    elif offset < 0:
                        command = 'a'
                    else:
                        command = 'd'
            else:
                command = 'x'

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