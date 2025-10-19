# LLFF to PyTorch3D Dataset Converter

A toolkit for converting raw LLFF (Local Light Field Fusion) datasets to PyTorch3D educational format for NeRF training.

## 🚀 Quick Start

### 1. Download Dataset from Kaggle

**📥 Download from: https://www.kaggle.com/datasets/arenagrenade/llff-dataset-full**

**Option A: Using Kaggle API (Recommended)**
```bash
# Install Kaggle API
pip install kagglehub

# Set up credentials (get API key from https://www.kaggle.com/account)
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset to nerf/ folder
python download_llff.py
```

**Option B: Manual Download**
1. Go to: **https://www.kaggle.com/datasets/arenagrenade/llff-dataset-full**
2. Click "Download" button (requires Kaggle account)
3. Extract the downloaded file to `nerf/data/nerf_llff_data/` folder
4. Ensure the folder structure looks like: `nerf/data/nerf_llff_data/fern/`, `nerf/data/nerf_llff_data/flower/`, etc.

### 2. Convert All Scenes
```bash
# Make script executable
chmod +x preprocess_all_scenes.sh

# Convert all 8 LLFF scenes to PyTorch3D format
./preprocess_all_scenes.sh
```

**Note:** Before running, update the paths in `preprocess_all_scenes.sh`:
```bash
# Edit the script to set your actual paths
INPUT_DIR="/your/actual/path/LLFF_blender_to_pytorch_dataset/nerf/data/nerf_llff_data"
OUTPUT_DIR="/your/actual/path/LLFF_blender_to_pytorch_dataset/output"
```

### 3. Use Converted Data
```python
import torch
from PIL import Image

# Load any scene
data = torch.load('output/fern.pth', weights_only=False)
images = Image.open('output/fern.png')
```

## 📁 Project Structure

```
/path/to/LLFF_blender_to_pytorch_dataset/
├── README.md                           # This guide
├── .gitignore                          # Excludes nerf/ and output/ folders
├── download_llff.py                    # Kaggle dataset downloader
├── preprocess_llff_to_pytorch3d.py    # Main converter
├── preprocess_blender_to_pytorch3d.py # Blender converter
├── preprocess_all_scenes.sh           # Batch processing
├── example_usage.py                   # Usage examples
├── nerf/                              # Raw datasets (gitignored)
│   └── data/nerf_llff_data/           # LLFF scenes
└── output/                            # Converted data (gitignored)
    ├── fern.pth, fern.png
    ├── flower.pth, flower.png
    └── ...
```

## 📊 Available Scenes

| Scene    | Images | Description |
|----------|--------|-------------|
| 🌿 Fern  | 20     | Natural fern plant |
| 🌸 Flower| 34     | Colorful flower |
| 🏰 Fortress| 42   | Stone fortress |
| 🦌 Horns | 62     | Animal horns |
| 🍃 Leaves| 26     | Tree leaves |
| 🌺 Orchids| 25    | Orchid flowers |
| 🏠 Room  | 41     | Indoor room |
| 🦕 T-Rex | 55     | Dinosaur skeleton |

**Total:** 305 images across 8 scenes

## 🔧 Installation

```bash
# Install dependencies
pip install torch torchvision pytorch3d numpy pillow imageio kagglehub
```

## 📖 Usage

### Convert Single Scene
```bash
python preprocess_llff_to_pytorch3d.py \
    --scene_dir nerf/data/nerf_llff_data/fern \
    --out output \
    --width 504 \
    --height 378
```

### Convert Blender Scene
```bash
python preprocess_blender_to_pytorch3d.py \
    --scene_dir nerf/data/nerf_synthetic/lego \
    --out output
```

## 📄 Data Format

**Input (LLFF):**
```
scene_directory/
├── images/                    # JPG/PNG images
├── poses_bounds.npy          # Camera poses (N, 17)
└── sparse/                   # COLMAP data
```

**Output (PyTorch3D):**
```
output/
├── scene_name.pth            # Camera data + metadata
└── scene_name.png            # Concatenated images
```

**`.pth` file contents:**
```python
{
    "cameras": {
        "R": torch.Tensor,              # (N, 3, 3) rotation matrices
        "T": torch.Tensor,              # (N, 3) translation vectors
        "focal_length": torch.Tensor,   # (N, 2) focal lengths
        "principal_point": torch.Tensor # (N, 2) principal points
    },
    "split": (train_idx, val_idx, test_idx),  # 80/10/10 split
    "bounds": np.array,                 # (N, 2) near/far bounds
    "image_size": [height, width]
}
```

## 🐛 Troubleshooting

1. **"No images found"** - Check images/ directory exists with .jpg/.png files
2. **"poses_bounds.npy not found"** - Ensure file exists in scene directory
3. **Memory issues** - Use smaller width/height for resizing
4. **Kaggle API errors** - Verify ~/.kaggle/kaggle.json format and permissions

## 📈 Performance

| Scene    | Images | Processing Time | Output Size |
|----------|--------|----------------|-------------|
| Fern     | 20     | ~30 seconds    | 6.1 MB      |
| Flower   | 34     | ~45 seconds    | 11.3 MB     |
| Horns    | 62     | ~90 seconds    | 18.4 MB     |

## 📞 Support

For detailed documentation, see `README_preprocessing.txt`.

---

**Happy NeRF training! 🚀**