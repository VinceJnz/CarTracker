import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import sys
import os
import cv2

# Add the PSMNet directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def resize_to_divisible(img, divisor=16):
    """Resize image to make dimensions divisible by divisor"""
    h, w = img.shape[:2]
    new_h = h - (h % divisor)
    new_w = w - (w % divisor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def load_psmnet(model_path=None, maxdisp=96):
    from models.stackhourglass import PSMNet
    
    model = PSMNet(maxdisp=maxdisp).to(device)
    model = nn.DataParallel(model)
    
    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])
    
    model.eval()
    return model

def preprocess_stereo_pair(left_img, right_img):
    # Convert to uint8 if needed
    if left_img.dtype == np.float32:
        left_img = (left_img * 255).astype(np.uint8)
        right_img = (right_img * 255).astype(np.uint8)
    
    # Resize to divisible dimensions
    left_img = resize_to_divisible(left_img)
    right_img = resize_to_divisible(right_img)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    left_tensor = transform(left_img).unsqueeze(0).to(device)
    right_tensor = transform(right_img).unsqueeze(0).to(device)
    
    return left_tensor, right_tensor, left_img.shape[:2]

if __name__ == "__main__":
    # Initialize model
    model = load_psmnet("pretrained_sceneflow.tar", maxdisp=96)
    
    # Load demo images
    left_path = "/app/data/stereo/im2.png"
    right_path = "/app/data/stereo/im6.png"
    
    left_img = cv2.imread(left_path)[:,:,::-1]  # BGR to RGB
    right_img = cv2.imread(right_path)[:,:,::-1]
    
    if left_img is None or right_img is None:
        raise FileNotFoundError(f"Could not load images. Check paths:\n{left_path}\n{right_path}")
    
    # Preprocess and get original dimensions
    left, right, original_shape = preprocess_stereo_pair(left_img, right_img)
    
    # Inference
    with torch.no_grad():
        disparity = model(left, right)
    
    # Process disparity
    disparity_np = disparity[0].cpu().numpy().squeeze()
    
    # Resize disparity to match original image dimensions
    disp_vis = cv2.resize(disparity_np, (original_shape[1], original_shape[0]))
    
    # Normalize for visualization
    disp_vis = (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min())
    disp_vis = (disp_vis * 255).astype(np.uint8)
    
    # Create output directory
    output_dir = "/app/data/stereo/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save outputs
    cv2.imwrite(f'{output_dir}/disparity.png', disp_vis)
    np.save(f'{output_dir}/disparity.npy', disparity_np)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imwrite(f'{output_dir}/disparity_heatmap.png', heatmap)
    
    # Create comparison image (now with matching dimensions)
    h, w = original_shape
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
    comparison[:, :w] = cv2.resize(left_img, (w, h))
    comparison[:, w:w*2] = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2RGB)
    comparison[:, w*2:] = heatmap
    cv2.imwrite(f'{output_dir}/comparison.png', comparison)
    
    print(f"Successfully processed and saved results to {output_dir}/")