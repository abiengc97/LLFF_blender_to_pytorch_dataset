import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch

def stack_images_vertically(image_paths, target_size=None):
    """
    Load images, (optionally) resize to target_size=(W,H), and stack vertically.
    Returns (H_total, W, 3) float array in [0,1].
    """
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

    big = np.concatenate(imgs, axis=0)  # vertical stack
    return big.astype(np.float32) / 255.0, (W0, H0)

def convert_llff_scene(scene_dir, output_dir="./data", scene_name=None, width=None, height=None):
    """
    Convert a single LLFF scene directory (with images/ + poses_bounds.npy) into PyTorch3D format.
    - scene_dir: path to e.g., /data/llff/fern
    - output_dir: where to write <scene>.pth and <scene>.png
    - width,height: optional resize. If omitted, uses original size.
    """
    scene_dir = os.path.abspath(scene_dir)
    if scene_name is None:
        scene_name = os.path.basename(os.path.normpath(scene_dir))

    poses_bounds_path = os.path.join(scene_dir, "poses_bounds.npy")
    if not os.path.isfile(poses_bounds_path):
        raise FileNotFoundError(f"poses_bounds.npy not found at {poses_bounds_path}")

    # Load LLFF poses & bounds
    poses_bounds = np.load(poses_bounds_path)  # (N, 17) typically
    poses = poses_bounds[:, :-2].reshape(-1, 3, 5)  # (N, 3, 5)
    bounds = poses_bounds[:, -2:]  # near/far per image (unused here but kept for reference)

    # Intrinsics are stored in the last column: [H, W, focal]
    H, W, focal = poses[0, :, -1]
    H, W, focal = float(H), float(W), float(focal)

    # Camera extrinsics: (3x4) per image in poses[..., :4]
    c2w_list = poses[:, :, :4]  # (N, 3, 4)

    # Build image list
    image_paths = sorted(
        glob.glob(os.path.join(scene_dir, "images", "*.png"))
        + glob.glob(os.path.join(scene_dir, "images", "*.jpg"))
        + glob.glob(os.path.join(scene_dir, "images", "*.jpeg"))
        + glob.glob(os.path.join(scene_dir, "images", "*.PNG"))
        + glob.glob(os.path.join(scene_dir, "images", "*.JPG"))
        + glob.glob(os.path.join(scene_dir, "images", "*.JPEG"))
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found under {os.path.join(scene_dir, 'images')}")

    n = len(image_paths)
    if n != c2w_list.shape[0]:
        print(f"Warning: number of images ({n}) != number of poses ({c2w_list.shape[0]}). Using the min.")
        n = min(n, c2w_list.shape[0])
        image_paths = image_paths[:n]
        c2w_list = c2w_list[:n]

    # Optional resize
    if width is not None and height is not None:
        target_size = (int(width), int(height))
        sx, sy = target_size[0] / W, target_size[1] / H
        W_new, H_new = target_size
        focal_x = focal * sx
        focal_y = focal * sy
        pp_x, pp_y = (W_new / 2.0), (H_new / 2.0)
    else:
        target_size = None
        W_new, H_new = int(W), int(H)
        focal_x = focal_y = focal
        pp_x, pp_y = (W / 2.0), (H / 2.0)

    # Stack images into one big PNG (vertical)
    big_img, (W_final, H_final) = stack_images_vertically(image_paths, target_size)
    assert W_final == W_new and H_final == H_new, "Unexpected final image size mismatch."

    # Compute world-to-camera extrinsics R, T from LLFF c2w
    # c2w is [R_c2w | t_c2w], so w2c = inverse(c2w_4x4)
    R_list, T_list = [], []
    for i in range(n):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :4] = c2w_list[i]
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        R_list.append(torch.tensor(R, dtype=torch.float32))
        T_list.append(torch.tensor(T, dtype=torch.float32))

    R = torch.stack(R_list, dim=0)             # (N, 3, 3)
    T = torch.stack(T_list, dim=0)             # (N, 3)
    focal_length = torch.tensor([[focal_x, focal_y]], dtype=torch.float32).repeat(n, 1)
    principal_point = torch.tensor([[pp_x, pp_y]], dtype=torch.float32).repeat(n, 1)

    # Simple 80/10/10 split
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    train_idx = list(range(0, n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, n))

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    # Save big image
    big_uint8 = (np.clip(big_img, 0.0, 1.0) * 255.0).astype(np.uint8)
    big_save_path = os.path.join(output_dir, f"{scene_name}.png")
    Image.fromarray(big_uint8).save(big_save_path)

    # Save cameras & split
    data = {
        "cameras": {
            "R": R,
            "T": T,
            "focal_length": focal_length,
            "principal_point": principal_point,
        },
        "split": (train_idx, val_idx, test_idx),
        # Optional: include bounds for future use
        "bounds": bounds[:n].astype(np.float32),
        "image_size": [H_new, W_new],
        "note": "Converted from LLFF (poses_bounds.npy) into PyTorch3D NeRF format."
    }
    torch.save(data, os.path.join(output_dir, f"{scene_name}.pth"))

    print(f"[OK] Wrote: {big_save_path}")
    print(f"[OK] Wrote: {os.path.join(output_dir, scene_name + '.pth')}")
    print(f"Images: {n} | Size: {H_new}x{W_new} | Focal: ({focal_x:.2f}, {focal_y:.2f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="Path to LLFF scene folder (contains images/ and poses_bounds.npy)")
    ap.add_argument("--out", default="./data", help="Output directory for <scene>.pth and <scene>.png")
    ap.add_argument("--width", type=int, default=None, help="Optional resize width (pixels)")
    ap.add_argument("--height", type=int, default=None, help="Optional resize height (pixels)")
    ap.add_argument("--scene_name", type=str, default=None, help="Optional override for output scene name")
    args = ap.parse_args()
    convert_llff_scene(args.scene_dir, args.out, args.scene_name, args.width, args.height)