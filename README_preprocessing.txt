LLFF Dataset Preprocessing Guide
=====================================

This guide explains how to convert raw LLFF datasets to PyTorch3D educational format using the preprocess_llff_to_pytorch3d.py script.

OVERVIEW
--------
The preprocessing script converts raw LLFF datasets (JPG images + poses_bounds.npy) into educational PyTorch3D format (.pth + .png files) suitable for NeRF training.

INPUT FORMAT
------------
Raw LLFF dataset structure:
```
scene_directory/
├── images/                    # Original JPG/PNG images
│   ├── IMG_0001.JPG
│   ├── IMG_0002.JPG
│   └── ...
├── poses_bounds.npy          # Camera poses and bounds (N, 17)
└── other_files/              # Optional COLMAP files
```

OUTPUT FORMAT
-------------
Educational PyTorch3D format:
```
output_directory/
├── scene_name.pth            # Camera data and metadata
└── scene_name.png            # Concatenated images (vertical stack)
```

SCRIPT USAGE
============

Basic Command:
--------------
python preprocess_llff_to_pytorch3d.py --scene_dir <path_to_scene> --out <output_directory>

Required Arguments:
------------------
--scene_dir    Path to LLFF scene folder (contains images/ and poses_bounds.npy)

Optional Arguments:
------------------
--out          Output directory for .pth and .png files (default: ./data)
--width        Resize width in pixels (default: None, uses original)
--height       Resize height in pixels (default: None, uses original)
--scene_name   Override output scene name (default: folder name)

EXAMPLES
========

1. Convert single scene (original size):
----------------------------------------
python preprocess_llff_to_pytorch3d.py --scene_dir /path/to/fern --out ./output

2. Convert single scene with resizing:
-------------------------------------
python preprocess_llff_to_pytorch3d.py --scene_dir /path/to/fern --out ./output --width 504 --height 378

3. Convert with custom scene name:
---------------------------------
python preprocess_llff_to_pytorch3d.py --scene_dir /path/to/fern --out ./output --scene_name my_fern

BATCH CONVERSION
================

Convert all scenes in a dataset:
--------------------------------
#!/bin/bash
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
input_dir="/path/to/nerf_llff_data"
output_dir="./output"

for scene in "${scenes[@]}"; do
    echo "Converting $scene..."
    python preprocess_llff_to_pytorch3d.py \
        --scene_dir "$input_dir/$scene" \
        --out "$output_dir" \
        --width 504 \
        --height 378 \
        --scene_name "$scene"
done

Python batch script:
-------------------
import os
import subprocess

scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
input_dir = "/path/to/nerf_llff_data"
output_dir = "./output"

for scene in scenes:
    cmd = [
        "python", "preprocess_llff_to_pytorch3d.py",
        "--scene_dir", f"{input_dir}/{scene}",
        "--out", output_dir,
        "--width", "504",
        "--height", "378",
        "--scene_name", scene
    ]
    print(f"Converting {scene}...")
    subprocess.run(cmd)

OUTPUT DETAILS
==============

.pth File Contents:
-------------------
{
    "cameras": {
        "R": torch.Tensor,              # (N, 3, 3) rotation matrices
        "T": torch.Tensor,              # (N, 3) translation vectors
        "focal_length": torch.Tensor,   # (N, 2) focal lengths (fx, fy)
        "principal_point": torch.Tensor # (N, 2) principal points (cx, cy)
    },
    "split": (train_idx, val_idx, test_idx),  # 80/10/10 split
    "bounds": np.array,                 # (N, 2) near/far bounds
    "image_size": [height, width],      # Final image dimensions
    "note": "Converted from LLFF format"
}

.png File:
----------
- All images concatenated vertically
- Shape: (total_height, width, 3)
- total_height = num_images * individual_height
- Values: 0-255 uint8

USAGE IN CODE
=============

Loading the converted data:
---------------------------
import torch
from PIL import Image

# Load camera data
data = torch.load('scene.pth', weights_only=False)
R = data['cameras']['R']           # (N, 3, 3)
T = data['cameras']['T']           # (N, 3)
focal = data['cameras']['focal_length']  # (N, 2)
principal = data['cameras']['principal_point']  # (N, 2)

# Load images
images = Image.open('scene.png')    # PIL Image
images_array = np.array(images)     # (H_total, W, 3)

# Get train/val/test splits
train_idx, val_idx, test_idx = data['split']

TROUBLESHOOTING
===============

Common Issues:
--------------

1. "No images found" error:
   - Check that images/ directory exists
   - Verify image file extensions (.jpg, .JPG, .png, .PNG)
   - Script supports both lowercase and uppercase extensions

2. "poses_bounds.npy not found" error:
   - Ensure poses_bounds.npy exists in scene directory
   - Check file permissions

3. Image/pose count mismatch:
   - Script will use minimum of image count and pose count
   - Warning will be printed

4. Memory issues with large datasets:
   - Use smaller width/height for resizing
   - Process scenes individually

PERFORMANCE NOTES
=================

- Processing time scales with number of images
- Resizing adds computational overhead
- Large scenes (60+ images) may take several minutes
- Output file sizes depend on image count and resolution

TYPICAL SCENE SIZES
===================

| Scene    | Images | Original Size | 504x378 Size |
|----------|--------|---------------|--------------|
| fern     | 20     | ~6MB          | ~6MB         |
| flower   | 34     | ~12MB         | ~12MB        |
| fortress | 42     | ~13MB         | ~13MB        |
| horns    | 62     | ~18MB         | ~18MB        |
| leaves   | 26     | ~11MB         | ~11MB        |
| orchids  | 25     | ~10MB         | ~10MB        |
| room     | 41     | ~7MB          | ~7MB         |
| trex     | 55     | ~15MB         | ~15MB        |

EDUCATIONAL BENEFITS
====================

- Simplified loading: Just 2 files per scene
- Consistent format: Same structure across all scenes
- PyTorch3D compatible: Works with educational NeRF implementations
- Self-contained: No external dependencies
- Ready to use: No additional preprocessing needed

This preprocessing makes LLFF datasets much more accessible for educational purposes while maintaining all the essential data needed for NeRF training.
