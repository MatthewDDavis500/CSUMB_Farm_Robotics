import cv2
import torch
import numpy as np
import socket
from torchvision import transforms
import depthai as dai
import segmentation_models_pytorch as smp
from PIL import Image

# TCP Settings
ROBOT_IP = "100.87.161.11"
PORT = 9999

# Load UNet ResNet34 model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load("unet_resnet34Final.pth", map_location=torch.device("cpu")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Image transform (standard normalization for ResNet34)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# DepthAI pipeline setup
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

def enhance_input_image(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE and sharpening to enhance contrast and edges."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply sharpening kernel
    sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_bgr, -1, sharpen_kernel)

    return sharpened

def determine_command_from_mask(mask: np.ndarray) -> str:
    """Returns 'w', 'a', 'd', or 'x' based on path position in the mask."""
    h, w = mask.shape
    roi = mask[h//2:, :]
    coords = np.column_stack(np.where(roi > 0.5))
    
    if coords.size == 0:
        return "x"  # No path visible → stop

    avg_x = np.mean(coords[:, 1])
    center_x = w / 2
    offset = avg_x - center_x

    # Define pixel thresholds to avoid noise
    tolerance = 30  # pixels left/right of center is still "straight"
    if abs(offset) < tolerance:
        return "w"
    elif offset < 0:
        return "a"
    else:
        return "d"

def send_command(sock: socket.socket, cmd: str):
    """Send w/a/s/d/x command as a single character."""
    try:
        sock.sendall(cmd.encode())
        print(f"[COMMAND] Sent: {cmd}")
    except Exception as e:
        print(f"[TCP ERROR] {e}")

def main():
    with dai.Device(pipeline) as dai_device, \
         socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        print(f"[Planner] Connecting to controller at {ROBOT_IP}:{PORT} ...")
        s.connect((ROBOT_IP, PORT))
        print("[Planner] Connected!")

        video_queue = dai_device.getOutputQueue(name="video", maxSize=4, blocking=False)

        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()

            # --- Preprocess input to improve segmentation ---
            enhanced_frame = enhance_input_image(frame)

            # Convert to PIL Image and prepare input tensor
            input_pil = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(input_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                output = torch.sigmoid(output)
                mask = (output > 0.5).float()
                mask_np = mask.squeeze().cpu().numpy()

            # Decide on command
            command = determine_command_from_mask(mask_np)
            send_command(s, command)

            # Show overlay
            color_mask = (mask_np * 255).astype(np.uint8)
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
            overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
            cv2.imshow("Segmented View", overlay)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()