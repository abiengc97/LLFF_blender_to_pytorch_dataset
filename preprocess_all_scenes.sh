#!/bin/bash

# LLFF Dataset Preprocessing Script
# Converts all LLFF scenes to PyTorch3D educational format

# Configuration
INPUT_DIR="/home/agcheria/LLFF_to_pytorch_dataset/nerf/data/nerf_llff_data"
OUTPUT_DIR="/home/agcheria/LLFF_to_pytorch_dataset/output"
WIDTH=504
HEIGHT=378

# List of all LLFF scenes
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

echo "=========================================="
echo "LLFF Dataset Preprocessing Script"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Target size: ${WIDTH}x${HEIGHT}"
echo "Scenes to process: ${#SCENES[@]}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each scene
for scene in "${SCENES[@]}"; do
    scene_dir="$INPUT_DIR/$scene"
    
    if [ ! -d "$scene_dir" ]; then
        echo "❌ Scene directory not found: $scene_dir"
        continue
    fi
    
    echo "🌿 Processing $scene scene..."
    echo "   Input: $scene_dir"
    echo "   Output: $OUTPUT_DIR/${scene}.pth + ${scene}.png"
    
    # Run preprocessing
    python preprocess_llff_to_pytorch3d.py \
        --scene_dir "$scene_dir" \
        --out "$OUTPUT_DIR" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --scene_name "$scene"
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $scene converted successfully"
    else
        echo "   ❌ $scene conversion failed"
    fi
    echo ""
done

echo "=========================================="
echo "Preprocessing Complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Files created:"
ls -la "$OUTPUT_DIR"/*.pth "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l | xargs echo "Total files:"
echo ""
echo "Usage example:"
echo "  import torch"
echo "  data = torch.load('$OUTPUT_DIR/fern.pth', weights_only=False)"
echo "  images = Image.open('$OUTPUT_DIR/fern.png')"
