#!/usr/bin/env python3
"""
Script to download the LLFF dataset from Kaggle and convert it to PyTorch3D format.
This addresses the issue with kagglehub.load_dataset() expecting tabular data.
"""

import kagglehub
import os
import numpy as np
import imageio
from PIL import Image
import torch

def download_llff_dataset(download_path):
    """Download the LLFF dataset from Kaggle."""
    print("Downloading LLFF dataset from Kaggle...")
    
    # Create directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download the dataset
    dataset_path = kagglehub.dataset_download(
        "arenagrenade/llff-dataset-full",
        path=download_path
    )
    
    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path

def list_dataset_contents(dataset_path):
    """List the contents of the downloaded dataset."""
    print("\nDataset contents:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

def convert_llff_to_pytorch3d(llff_path, output_path):
    """Convert LLFF dataset to PyTorch3D format."""
    print("\nConverting LLFF to PyTorch3D format...")
    
    # Find fern dataset
    fern_path = None
    for root, dirs, files in os.walk(llff_path):
        if 'fern' in root.lower():
            fern_path = root
            break
    
    if not fern_path:
        print("Fern dataset not found!")
        return
    
    print(f"Found fern dataset at: {fern_path}")
    
    # Load poses
    poses_file = os.path.join(fern_path, 'poses_bounds.npy')
    if not os.path.exists(poses_file):
        print(f"Poses file not found: {poses_file}")
        return
    
    poses = np.load(poses_file)
    print(f"Loaded poses: {poses.shape}")
    
    # Load images
    images_dir = os.path.join(fern_path, 'images')
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.JPG', '.jpg', '.png'))])
    print(f"Found {len(image_files)} images")
    
    # Process poses (convert from LLFF format to PyTorch3D format)
    poses_reshaped = poses[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])  # (3, 5, 20)
    bds = poses[:, -2:].transpose([1,0])  # (2, 20)
    
    # Extract camera parameters
    R = poses_reshaped[:3, :3, :].transpose(2, 0, 1)  # (20, 3, 3)
    T = poses_reshaped[:3, 3, :].T  # (20, 3)
    focal_length = poses_reshaped[:3, 4, :].T  # (20, 3)
    
    # Convert focal length to (fx, fy) format
    focal_length_2d = focal_length[:, :2]  # (20, 2)
    
    # Create principal points (assume center of image)
    # We'll need to load an image to get the dimensions
    sample_img = imageio.imread(os.path.join(images_dir, image_files[0]))
    h, w = sample_img.shape[:2]
    principal_point = np.array([[w/2, h/2]] * len(image_files))  # (20, 2)
    
    # Create train/val/test splits (same as original NeRF)
    n_images = len(image_files)
    train_idx = [i for i in range(n_images) if i not in [0, 8, 16]]
    val_idx = [0, 8, 16]
    test_idx = [0, 8, 16]
    
    # Save camera data
    camera_data = {
        'cameras': {
            'R': torch.from_numpy(R).float(),
            'T': torch.from_numpy(T).float(),
            'focal_length': torch.from_numpy(focal_length_2d).float(),
            'principal_point': torch.from_numpy(principal_point).float()
        },
        'split': [torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)]
    }
    
    # Save images as concatenated array
    images = []
    for img_file in image_files:
        img = imageio.imread(os.path.join(images_dir, img_file))
        # Resize to standard size (378x504) like in the original preprocessing
        img_resized = Image.fromarray(img).resize((504, 378))
        images.append(np.array(img_resized))
    
    # Concatenate images vertically
    images_concat = np.concatenate(images, axis=0)  # (7560, 504, 3)
    
    # Save data
    os.makedirs(output_path, exist_ok=True)
    
    # Save camera data
    torch.save(camera_data, os.path.join(output_path, 'fern.pth'))
    
    # Save concatenated images
    Image.fromarray(images_concat).save(os.path.join(output_path, 'fern.png'))
    
    print(f"Converted dataset saved to: {output_path}")
    print(f"Camera data: {os.path.join(output_path, 'fern.pth')}")
    print(f"Images: {os.path.join(output_path, 'fern.png')}")

if __name__ == "__main__":
    # Set paths
    download_path = "/home/agcheria/LLFF_to_pytorch_dataset/nerf/data/nerf_llff_data"
    output_path = "/home/agcheria/LLFF_to_pytorch_dataset/pytorch3d_data"
    
    # Download dataset
    dataset_path = download_llff_dataset(download_path)
    
    # List contents
    list_dataset_contents(dataset_path)
    
    # Convert to PyTorch3D format
    convert_llff_to_pytorch3d(dataset_path, output_path)
    
    print("\nDone! You can now use the converted dataset with PyTorch3D.")
