import argparse
import cv2
import numpy as np
import onnxruntime as ort

# Run ONNX file with test image.
# Save visualized image 
def run(stereo_image_name, width, height, onnx_path, output_name):
    # Load input image.
    # Input images are joined horizontally into a single image.
    img = cv2.imread(stereo_image_name)[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img0 = img[:, : w // 2, :]
    img1 = img[:, w // 2 :, :]

    # Resize to model input size
    img0 = cv2.resize(img0, (width, height))
    img1 = cv2.resize(img1, (width, height))

    # HWC uint8 -> NCHW float32
    # values in 0-255 (model normalizes internally).
    left = img0.astype(np.float32).transpose(2, 0, 1)[None]
    right = img1.astype(np.float32).transpose(2, 0, 1)[None]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    disp = sess.run(["disparity"], {"left": left, "right": right})[0]
    disp = disp[0, 0]  # (H, W)

    # Save sample output 
    print("Disparity shape:", disp.shape, "min:", disp.min(), "max:", disp.max())

    vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6) * 255
    vis = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(output_name, vis)
    print("Saved", output_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX model on a test image.")
    parser.add_argument("--input", default="./assets/Explorer_HD2K_SN28883284_20-42-06.png", help="left+right stereo image")
    parser.add_argument("--onnx_path", default="liteanystereo.onnx", help="ONNX file")
    parser.add_argument("--output", default="disp_vis.png", help="output visualized file")
    parser.add_argument("--width", type=int, default=1248, help="per input, not left+right")
    parser.add_argument("--height", type=int, default=384)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(stereo_image_name=args.input,
        width=args.width,
        height=args.height,
        onnx_path=args.onnx_path,
        output_name=args.output)
