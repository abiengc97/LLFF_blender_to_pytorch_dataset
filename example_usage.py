#!/usr/bin/env python3
"""
Example usage of the preprocessed LLFF datasets
This script demonstrates how to load and use the converted PyTorch3D format data
"""

import torch
import numpy as np
from PIL import Image
import os

def load_scene_data(scene_name, data_dir="/home/agcheria/LLFF_to_pytorch_dataset/output"):
    """
    Load a preprocessed scene from the output directory
    
    Args:
        scene_name (str): Name of the scene (e.g., 'fern', 'flower')
        data_dir (str): Directory containing the preprocessed data
    
    Returns:
        dict: Dictionary containing camera data and image information
    """
    pth_path = os.path.join(data_dir, f"{scene_name}.pth")
    png_path = os.path.join(data_dir, f"{scene_name}.png")
    
    if not os.path.exists(pth_path) or not os.path.exists(png_path):
        raise FileNotFoundError(f"Scene data not found for {scene_name}")
    
    # Load camera data
    data = torch.load(pth_path, weights_only=False)
    
    # Load concatenated images
    images = Image.open(png_path)
    images_array = np.array(images)  # (H_total, W, 3)
    
    # Calculate individual image dimensions
    n_images = data['cameras']['R'].shape[0]
    h, w = data['image_size']
    
    # Split concatenated images back into individual images
    individual_images = []
    for i in range(n_images):
        start_h = i * h
        end_h = (i + 1) * h
        img = images_array[start_h:end_h, :, :]  # (H, W, 3)
        individual_images.append(img)
    
    return {
        'cameras': data['cameras'],
        'split': data['split'],
        'bounds': data['bounds'],
        'image_size': data['image_size'],
        'individual_images': individual_images,
        'concatenated_images': images_array,
        'n_images': n_images
    }

def print_scene_info(scene_data, scene_name):
    """Print information about the loaded scene"""
    print(f"\n=== {scene_name.upper()} SCENE INFO ===")
    print(f"Number of images: {scene_data['n_images']}")
    print(f"Image size: {scene_data['image_size'][0]}x{scene_data['image_size'][1]}")
    print(f"Concatenated image shape: {scene_data['concatenated_images'].shape}")
    
    # Camera info
    R = scene_data['cameras']['R']
    T = scene_data['cameras']['T']
    focal = scene_data['cameras']['focal_length']
    principal = scene_data['cameras']['principal_point']
    
    print(f"Camera rotation shape: {R.shape}")
    print(f"Camera translation shape: {T.shape}")
    print(f"Focal length shape: {focal.shape}")
    print(f"Principal point shape: {principal.shape}")
    
    # Split info
    train_idx, val_idx, test_idx = scene_data['split']
    print(f"Train images: {len(train_idx)}")
    print(f"Validation images: {len(val_idx)}")
    print(f"Test images: {len(test_idx)}")
    
    # Sample camera parameters
    print(f"\nSample camera parameters (first camera):")
    print(f"  Rotation R[0]:\n{R[0]}")
    print(f"  Translation T[0]: {T[0]}")
    print(f"  Focal length: {focal[0]}")
    print(f"  Principal point: {principal[0]}")

def main():
    """Main function demonstrating usage"""
    print("LLFF Dataset Usage Example")
    print("=" * 50)
    
    # Available scenes
    scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    
    print(f"Available scenes: {', '.join(scenes)}")
    
    # Example: Load fern scene
    try:
        print(f"\nLoading fern scene...")
        fern_data = load_scene_data("fern")
        print_scene_info(fern_data, "fern")
        
        # Example: Access individual images
        print(f"\nFirst individual image shape: {fern_data['individual_images'][0].shape}")
        print(f"Image data type: {fern_data['individual_images'][0].dtype}")
        print(f"Image value range: {fern_data['individual_images'][0].min()} - {fern_data['individual_images'][0].max()}")
        
        # Example: Access camera data
        print(f"\nCamera data types:")
        print(f"  R: {fern_data['cameras']['R'].dtype}")
        print(f"  T: {fern_data['cameras']['T'].dtype}")
        print(f"  focal_length: {fern_data['cameras']['focal_length'].dtype}")
        print(f"  principal_point: {fern_data['cameras']['principal_point'].dtype}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run the preprocessing script first!")
    
    print(f"\n" + "=" * 50)
    print("Example complete!")

if __name__ == "__main__":
    main()
