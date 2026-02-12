import cv2
import depthai as dai
import numpy as np

def getFrame(queue):
    # Get frame from queue and convert to OpenCV format
    frame = queue.get()
    return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()
    
    # ✅ Use higher resolution for better long-range accuracy
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono.setBoardSocket(dai.CameraBoardSocket.LEFT if isLeft else dai.CameraBoardSocket.RIGHT)
    return mono

def getStereoPair(pipeline, monoLeft, monoRight):
    # Create and configure stereo node
    stereo = pipeline.createStereoDepth()
    
    # ✅ Improve long-range quality
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)  # Must be off for long-range
    stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    stereo.initialConfig.setConfidenceThreshold(180)  # Lower threshold = more data at distance

    # Link mono outputs to stereo inputs
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

if __name__ == "__main__":
    # Create pipeline
    pipeline = dai.Pipeline()

    # Set up mono cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    # Create stereo depth node
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    # XLink outputs
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.disparity.link(xoutDepth.input)

    xoutLeft = pipeline.createXLinkOut()
    xoutLeft.setStreamName("left")
    stereo.rectifiedLeft.link(xoutLeft.input)

    xoutRight = pipeline.createXLinkOut()
    xoutRight.setStreamName("right")
    stereo.rectifiedRight.link(xoutRight.input)

    # Start device
    with dai.Device(pipeline) as device:
        # Output queues
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

        # Disparity range 0-95 -> map to 0-255 for visualization
        disparityMultiplier = 255 / stereo.getMaxDisparity()

        sideBySide = True  # toggle key

        print("Press 't' to toggle stereo view style. Press 'q' to quit.")

        while True:
            # Get frames
            disparity = getFrame(qDepth)
            left = getFrame(qLeft)
            right = getFrame(qRight)

            # Normalize disparity for display
            disparity_visual = (disparity * disparityMultiplier).astype(np.uint8)
            disparity_colored = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)

            # Show stereo view and disparity
            if sideBySide:
                stereo_view = np.hstack((left, right))
                cv2.imshow("Stereo View", stereo_view)
            else:
                overlay = cv2.addWeighted(left, 0.5, right, 0.5, 0)
                cv2.imshow("Stereo View", overlay)

            cv2.imshow("Disparity", disparity_colored)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                sideBySide = not sideBySide

        cv2.destroyAllWindows()
