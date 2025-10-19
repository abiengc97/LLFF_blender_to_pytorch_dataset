import os
import json
import glob
import argparse
import numpy as np
from PIL import Image
import torch

def load_transforms(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def stack_images_vertically(image_paths, target_size=None):
    imgs = []
    W0 = H0 = None
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if target_size is not None:
            Wt, Ht = target_size
            img = img.resize((Wt, Ht), Image.BILINEAR)
        else:
            Wt, Ht = img.size
        if W0 is None:
            W0, H0 = Wt, Ht
        imgs.append(np.asarray(img))
    big = np.concatenate(imgs, axis=0)  # vertical
    return big.astype(np.float32) / 255.0, (W0, H0)

def invert_c2w(transform_matrix):
    # transform_matrix is 4x4 c2w; invert to get w2c
    c2w = np.array(transform_matrix, dtype=np.float32)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].astype(np.float32)
    T = w2c[:3, 3].astype(np.float32)
    return R, T

def convert_blender_scene(scene_dir, output_dir="./data", scene_name=None, width=None, height=None):
    """
    Convert a Blender NeRF scene with transforms_train/val/test.json into PyTorch3D format.
    """
    scene_dir = os.path.abspath(scene_dir)
    if scene_name is None:
        scene_name = os.path.basename(os.path.normpath(scene_dir))

    # Expect transforms_train/val/test.json
    json_train = os.path.join(scene_dir, "transforms_train.json")
    json_val   = os.path.join(scene_dir, "transforms_val.json")
    json_test  = os.path.join(scene_dir, "transforms_test.json")
    if not os.path.isfile(json_train):
        raise FileNotFoundError(f"Missing {json_train}")

    sets = []
    splits = []
    for jp, split_name in [(json_train, "train"), (json_val, "val"), (json_test, "test")]:
        if os.path.isfile(jp):
            d = load_transforms(jp)
            sets.append((d, split_name))
        else:
            sets.append((None, split_name))

    # Collect frames across splits
    frames, split_tags = [], []
    cam_angle_x = None
    for d, tag in sets:
        if d is None:
            continue
        if cam_angle_x is None:
            cam_angle_x = float(d.get("camera_angle_x"))
        for fr in d["frames"]:
            frames.append((fr, tag))
            split_tags.append(tag)

    if len(frames) == 0:
        raise RuntimeError("No frames found in transforms_*.json")

    # Determine size from the first image
    first_img_path = os.path.join(scene_dir, frames[0][0]["file_path"] + ".png")
    if not os.path.isfile(first_img_path):
        # try jpg
        first_img_path = os.path.join(scene_dir, frames[0][0]["file_path"] + ".jpg")
    if not os.path.isfile(first_img_path):
        raise FileNotFoundError(f"Cannot find image for first frame: {frames[0][0]['file_path']} (png/jpg)")

    W0, H0 = Image.open(first_img_path).size
    if width is not None and height is not None:
        Wt, Ht = int(width), int(height)
    else:
        Wt, Ht = W0, H0

    # Intrinsics from camera_angle_x: fx = 0.5*W / tan(0.5*angle_x)
    fx = 0.5 * Wt / np.tan(0.5 * cam_angle_x)
    fy = fx  # square pixels
    cx, cy = Wt / 2.0, Ht / 2.0

    # Gather images + extrinsics
    img_paths = []
    R_list, T_list = [], []
    train_idx, val_idx, test_idx = [], [], []
    idx = 0
    for fr, tag in frames:
        # Resolve file path relative to scene_dir
        rel = fr["file_path"]
        pngp = os.path.join(scene_dir, rel + ".png")
        imgp = pngp if os.path.isfile(pngp) else os.path.join(scene_dir, rel + ".jpg")
        if not os.path.isfile(imgp):
            raise FileNotFoundError(f"Missing image: {imgp}")
        img_paths.append(imgp)

        # Extrinsics
        R, T = invert_c2w(fr["transform_matrix"])
        R_list.append(torch.tensor(R))
        T_list.append(torch.tensor(T))

        # Split indices
        if tag == "train":
            train_idx.append(idx)
        elif tag == "val":
            val_idx.append(idx)
        elif tag == "test":
            test_idx.append(idx)
        idx += 1

    # Stack image strip
    big_img, (Wf, Hf) = stack_images_vertically(img_paths, target_size=(Wt, Ht))

    # Tensors
    n = len(img_paths)
    R = torch.stack(R_list, dim=0).float()
    T = torch.stack(T_list, dim=0).float()
    focal_length = torch.tensor([[fx, fy]], dtype=torch.float32).repeat(n, 1)
    principal_point = torch.tensor([[cx, cy]], dtype=torch.float32).repeat(n, 1)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    big_uint8 = (np.clip(big_img, 0.0, 1.0) * 255.0).astype(np.uint8)
    big_save_path = os.path.join(output_dir, f"{scene_name}.png")
    Image.fromarray(big_uint8).save(big_save_path)

    data = {
        "cameras": {
            "R": R,
            "T": T,
            "focal_length": focal_length,
            "principal_point": principal_point,
        },
        "split": (train_idx, val_idx, test_idx),
        "image_size": [Hf, Wf],
        "note": "Converted from Blender NeRF (transforms_*.json) into PyTorch3D NeRF format."
    }
    torch.save(data, os.path.join(output_dir, f"{scene_name}.pth"))
    print(f"[OK] Wrote: {big_save_path}")
    print(f"[OK] Wrote: {os.path.join(output_dir, scene_name + '.pth')}")
    print(f"Images: {n} | Size: {Hf}x{Wf} | Focal: ({fx:.2f}, {fy:.2f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="Path to Blender scene (with transforms_train/val/test.json)")
    ap.add_argument("--out", default="./data", help="Output directory for <scene>.pth and <scene>.png")
    ap.add_argument("--width", type=int, default=None, help="Optional resize width (pixels)")
    ap.add_argument("--height", type=int, default=None, help="Optional resize height (pixels)")
    ap.add_argument("--scene_name", type=str, default=None, help="Optional output scene name")
    args = ap.parse_args()
    convert_blender_scene(args.scene_dir, args.out, args.scene_name, args.width, args.height)