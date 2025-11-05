"""
sam2_segment.py

Script dùng để segment object trong ảnh bằng Segment Anything (SAM) và lưu kết quả.
Yêu cầu:
  - Python 3.8+
  - pip install segment-anything numpy pillow matplotlib
  - Tải checkpoint SAM (ví dụ: sam_vit_h_4b8939.pth) và chỉ định đường dẫn

Cách dùng:
  python sam2_segment.py --image path/to/image.jpg --checkpoint path/to/sam_checkpoint.pth --outdir outputs

Kết quả:
  - Lưu từng mask riêng dưới dạng PNG (b&w) trong thư mục outputs/masks
  - Lưu ảnh overlay (mask màu + alpha) trong outputs/overlays

Ghi chú: nếu bạn muốn dùng Hugging Face Sam2Processor/Sam2Model thay vì repo chính thức của Meta, báo mình biết — mình sẽ gửi phiên bản phù hợp.
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    # Thư viện chính thức Segment Anything (Meta)
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except Exception as e:
    raise ImportError(
        "Không thể import 'segment_anything'. Hãy cài đặt bằng:\n"
        "pip install segment-anything numpy pillow matplotlib\n"
        "Hoặc nếu đã cài, kiểm tra environment và python path.\n"
        f"Chi tiết lỗi: {e}"
    )


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def save_mask_png(mask_bool, outpath):
    # mask_bool: 2D boolean numpy array
    arr = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(arr).save(outpath)


def make_overlay(image_pil, mask_bool, alpha=0.4):
    """Tạo overlay: vẽ mask lên ảnh gốc với màu ngẫu nhiên và alpha."""
    img = image_pil.convert("RGBA")
    mask_img = Image.fromarray((mask_bool.astype(np.uint8)*255))

    # Tạo màu ngẫu nhiên cho mask
    color = tuple(np.random.randint(0, 256, size=3).tolist()) + (0,)
    color_layer = Image.new("RGBA", img.size, color)

    # Tạo alpha cho vùng mask
    mask_alpha = mask_img.convert("L").point(lambda p: int(p * alpha))
    color_layer.putalpha(mask_alpha)

    overlay = Image.alpha_composite(img, color_layer)
    return overlay


def run_mask_generation(image_path, checkpoint, outdir, model_type="vit_h", points_per_side=32, stability_score_thresh=None):
    # Tạo thư mục kết quả
    masks_out = os.path.join(outdir, "masks")
    overlays_out = os.path.join(outdir, "overlays")
    ensure_dir(masks_out)
    ensure_dir(overlays_out)

    # Load image
    img = load_image(image_path)
    img_arr = np.array(img)

    # Khởi tạo model SAM từ checkpoint
    print("Loading SAM model (this may take a while)...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    # Tạo mask generator tự động
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)

    print("Generating masks...")
    masks = mask_generator.generate(img_arr)
    print(f"Found {len(masks)} masks")

    # Lưu từng mask và overlay
    for i, m in enumerate(masks):
        seg = m.get("segmentation")
        if seg is None:
            continue

        # Lọc theo stability nếu cần (mỗi mask có key 'stability_score' tùy phiên bản)
        if stability_score_thresh is not None and (m.get("stability_score") is not None):
            if m.get("stability_score") < stability_score_thresh:
                continue

        mask_path = os.path.join(masks_out, f"mask_{i:03d}.png")
        save_mask_png(seg, mask_path)

        overlay = make_overlay(img, seg, alpha=0.35)
        overlay_path = os.path.join(overlays_out, f"overlay_{i:03d}.png")
        overlay.save(overlay_path)

    print("Done. Masks saved to:")
    print(" -", masks_out)
    print(" -", overlays_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment Anything (SAM) auto mask generator and saver")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--model-type", default="vit_h", help="SAM model type registered in sam_model_registry (e.g. vit_h, vit_l, vit_b)")
    parser.add_argument("--points-per-side", type=int, default=32, help="Controls density of automatic mask generator sampling")
    parser.add_argument("--stability-thresh", type=float, default=None, help="Optional filter: only save masks with stability_score >= threshold")

    args = parser.parse_args()

    run_mask_generation(
        image_path=args.image,
        checkpoint=args.checkpoint,
        outdir=args.outdir,
        model_type=args.model_type,
        points_per_side=args.points_per_side,
        stability_score_thresh=args.stability_thresh,
    )
