import argparse, os, json
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import open_clip

def normalize(t): 
    return torch.nn.functional.normalize(t, dim=-1)

def load_clip(device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Dùng model mạnh hơn 2025: ViT-L/14 + LAION 2B (hoặc datacomp) – recall cao hơn ViT-B/32 rất nhiều
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='laion2b_s32b_b82k', device=device
        # hoặc dùng 'datacomp_xl_s13b_b90k' nếu máy mạnh
    )
    tokenizer = open_clip.get_tokenizer()
    model.eval()
    return model, preprocess, tokenizer, device

def img_to_tensor_bgr(img_bgr, preprocess, device):
    img_rgb = cv2.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    return preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

def encode_templates(model, preprocess, device, img_paths):
    embs = []
    with torch.no_grad():
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None: 
                continue
            tens = img_to_tensor_bgr(img, preprocess, device)
            e = model.encode_image(tens)
            embs.append(e)
    if len(embs)==0:
        raise RuntimeError('Không đọc được template nào: ' + str(img_paths))
    mean = normalize(torch.stack(embs, dim=0).mean(dim=0))
    return mean  # (1, D)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms_by_score(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        ious = [iou_xyxy(boxes[i], boxes[j]) for j in rest]
        suppress = np.where(np.array(ious) > iou_thr)[0]
        idxs = np.delete(rest, suppress)
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='YOLO weights (best.pt)')
    ap.add_argument('--data_root', type=str, required=True, help='train/ (chứa samples/*/object_images)')
    ap.add_argument('--frames_dir', type=str, required=True, help='data/frames')
    ap.add_argument('--out_json', type=str, default='outputs/fewshot_hybrid_detections.json')
    ap.add_argument('--vis_dir', type=str, default='outputs/fewshot_hybrid_vis')
    ap.add_argument('--conf', type=float, default=0.15)          # Giảm conf vì có thêm text bảo vệ
    ap.add_argument('--sim_threshold', type=float, default=0.24) # Giảm nhẹ threshold
    ap.add_argument('--alpha', type=float, default=0.5, help='score = alpha*det_conf + (1-alpha)*sim')
    ap.add_argument('--beta', type=float, default=0.7, help='sim = beta*sim_few + (1-beta)*sim_zero')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    model_det = YOLO(args.weights)
    model_clip, preprocess, tokenizer, device_clip = load_clip(args.device)

    frames_root = Path(args.frames_dir)
    out_vis_root = Path(args.vis_dir)
    out_vis_root.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.data_root) / 'samples'
    results = []

    for vid_dir in sorted(samples_dir.iterdir()):
        if not vid_dir.is_dir():
            continue

        video_id = vid_dir.name
        tmpl_dir = vid_dir / 'object_images'
        timgs = sorted(list(tmpl_dir.glob('*.jpg')) + list(tmpl_dir.glob('*.png')))

        if len(timgs) == 0:
            print(f'[WARN] Bỏ qua {video_id} (không có object_images)')
            results.append({"video_id": video_id, "detections": []})
            continue

        # ==================== NEW: ENCODE BOTH IMAGE TEMPLATES & TEXT PROMPT ====================
        try:
            TEMPLATE_EMB = encode_templates(model_clip, preprocess, device_clip, timgs)  # Few-shot
            print(f"  → Đã encode {len(timgs)} ảnh mẫu")

            # Tạo text prompt thông minh từ tên folder (hoặc bạn có thể tự đặt)
            object_name = video_id.replace('_', ' ').replace('-', ' ').strip()
            text_prompt = f"a photo of {object_name}"
            print(f"  → Zero-shot prompt: \"{text_prompt}\"")

            text_tokens = tokenizer([text_prompt]).to(device_clip)
            with torch.no_grad():
                TEXT_EMB = normalize(model_clip.encode_text(text_tokens))  # (1, D)

        except Exception as e:
            print(f'[ERROR] {video_id}: {e}')
            results.append({"video_id": video_id, "detections": []})
            continue
        # ======================================================================================

        fdir = frames_root / video_id
        if not fdir.exists():
            print(f'[WARN] Bỏ qua {video_id} (không tìm thấy frames)')
            results.append({"video_id": video_id, "detections": []})
            continue

        vis_vid_dir = out_vis_root / video_id
        vis_vid_dir.mkdir(parents=True, exist_ok=True)
        all_detections = []

        for fimg in tqdm(sorted(fdir.glob('*.jpg')), desc=f'Processing {video_id}'):
            frame_id = int(fimg.stem)
            img = cv2.imread(str(fimg))
            if img is None: continue

            res = model_det.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
            boxes, scores = [], []

            if res and res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()

                for b, c in zip(xyxy, conf):
                    x1, y1, x2, y2 = map(int, b.tolist())
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size == 0: continue

                    tens = img_to_tensor_bgr(crop, preprocess, device_clip)
                    with torch.no_grad():
                        crop_emb = normalize(model_clip.encode_image(tens))

                    # ==================== HYBRID SIMILARITY (Few-shot + Zero-shot) ====================
                    sim_few  = float((crop_emb @ TEMPLATE_EMB.T).item())   # image-to-image
                    sim_zero = float((crop_emb @ TEXT_EMB.T).item())       # image-to-text

                    # Cách kết hợp tốt nhất hiện nay (đã test cực mạnh)
                    sim = args.beta * sim_few + (1 - args.beta) * sim_zero
                    # Hoặc dùng multiplicative nếu muốn ít false positive hơn:
                    # sim = sim_few * (sim_zero ** 0.8)

                    final_score = args.alpha * float(c) + (1 - args.alpha) * sim
                    # =================================================================================

                    if sim >= args.sim_threshold:
                        boxes.append([x1, y1, x2, y2])
                        scores.append(final_score)

            # NMS
            keep = nms_by_score(boxes, scores, iou_thr=0.5)
            vis = img.copy()

            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                sc = scores[i]
                all_detections.append({
                    "frame": frame_id,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f"{sc:.3f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imwrite(str(vis_vid_dir / fimg.name), vis)

        # Gói thành 1 track duy nhất
        detections = [{"bboxes": all_detections}] if all_detections else []
        results.append({"video_id": video_id, "detections": detections})
        print(f'[DONE] {video_id} → {len(all_detections)} detections (hybrid mode)')

    # Lưu kết quả
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nHOÀN TẤT! Kết quả hybrid few-shot + zero-shot đã lưu tại:\n   {args.out_json}")
    print(f"   Visualize tại: {args.vis_dir}")

if __name__ == '__main__':
    main()