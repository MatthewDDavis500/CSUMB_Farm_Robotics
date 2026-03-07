##########################################
#           HAS NO STOP FUNTIONS         #
#              FOR TESTING ONLY          #
##########################################


import cv2
import socket
import depthai as dai
from cvzone.PoseModule import PoseDetector

# TCP Settings
CONTROLLER_IP = "100.87.161.11"
CONTROLLER_PORT = 9999

# Setup socket connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((CONTROLLER_IP, CONTROLLER_PORT))
print(f"[Follower] Connected to controller at {CONTROLLER_IP}:{CONTROLLER_PORT}")

# Setup DepthAI pipeline for color camera
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Create pose detector
detector = PoseDetector()

# Connect to DepthAI device and start streaming
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    
    # Frame width for center calculations
    frame_width = 640
    frame_center = frame_width // 2
    center_tolerance = frame_width // 10  # Tolerance around center

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()

            # Use pose detector on the frame
            img = detector.findPose(frame)
            lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)

            if bboxInfo is not None and 'bbox' in bboxInfo:
                x, y, w, h = bboxInfo['bbox']
                cx = x + w // 2  # center x of bbox

                # Draw bounding box and center point
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)

                # Decide command based on horizontal position of bbox center
                offset = cx - frame_center
                if abs(offset) < center_tolerance:
                    command = 'w'  # Forward
                elif offset < 0:
                    command = 'a'  # Turn left
                else:
                    command = 'd'  # Turn right
            else:
                command = 'x'  # No person detected, stop

            # Send the command as a single char over TCP
            try:
                client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")

            cv2.imshow("Follower View", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Send stop command before exiting
                client_socket.sendall('x'.encode())
                break
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")