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
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    model.eval()
    return model, preprocess, device

def img_to_tensor_bgr(img_bgr, preprocess, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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

def get_slices(h, w, sh, sw, oh, ow):
    slices = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            x_end = min(x + sw, w)
            y_end = min(y + sh, h)
            x_start = max(0, x_end - sw)
            y_start = max(0, y_end - sh)
            slices.append((x_start, y_start, x_end, y_end))
            if x + sw >= w: break
            x += (sw - ow)
        if y + sh >= h: break
        y += (sh - oh)
    return slices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='YOLO weights (best.pt)')
    ap.add_argument('--data_root', type=str, required=True, help='train/ (chứa samples/*/object_images)')
    ap.add_argument('--frames_dir', type=str, required=True, help='data/frames')
    ap.add_argument('--out_json', type=str, default='outputs/fewshot_detections.json', help='Path to output JSON file')
    ap.add_argument('--vis_dir', type=str, default='outputs/fewshot_vis', help='Directory to save visualized frames')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--sim_threshold', type=float, default=0.28)
    ap.add_argument('--alpha', type=float, default=0.5, help='score = alpha*det_conf + (1-alpha)*sim')
    ap.add_argument('--slice_size', type=int, default=640, help='Size of slices for tiled inference')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    model_det = YOLO(args.weights)
    model_clip, preprocess, device_clip = load_clip(args.device)

    frames_root = Path(args.frames_dir)
    out_vis_root = Path(args.vis_dir)
    out_vis_root.mkdir(parents=True, exist_ok=True)

    samples_dir = Path(args.data_root) / 'samples'

    # Danh sách kết quả cuối cùng
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

        # Encode template
        try:
            TEMPLATE_EMB = encode_templates(model_clip, preprocess, device_clip, timgs)
        except Exception as e:
            print(f'[ERROR] {video_id}: {e}')
            results.append({"video_id": video_id, "detections": []})
            continue

        fdir = frames_root / video_id
        if not fdir.exists():
            print(f'[WARN] Bỏ qua {video_id} (không tìm thấy frames)')
            results.append({"video_id": video_id, "detections": []})
            continue

        # Thư mục visualize cho video này
        vis_vid_dir = out_vis_root / video_id
        vis_vid_dir.mkdir(parents=True, exist_ok=True)

        # Danh sách các bbox theo frame (sẽ được gom thành 1 track duy nhất như yêu cầu)
        all_detections = []  # list of dict: {"frame": int, "x1": int, ...}

        for fimg in tqdm(sorted(fdir.glob('*.jpg')), desc=f'Processing {video_id}'):
            frame_id = int(fimg.stem)
            img = cv2.imread(str(fimg))
            if img is None:
                continue

            # Slicing inference
            slice_wh = (args.slice_size, args.slice_size)
            overlap_wh = (int(args.slice_size * 0.2), int(args.slice_size * 0.2))
            slices = get_slices(img.shape[0], img.shape[1], slice_wh[0], slice_wh[1], overlap_wh[0], overlap_wh[1])

            boxes, scores = [], []

            for sx1, sy1, sx2, sy2 in slices:
                slice_img = img[sy1:sy2, sx1:sx2]
                if slice_img.size == 0: continue

                res = model_det.predict(slice_img, conf=args.conf, device=args.device, verbose=False)[0]
                
                if res and res.boxes is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    conf = res.boxes.conf.cpu().numpy()

                    for b, c in zip(xyxy, conf):
                        # Local coordinates in slice
                        lx1, ly1, lx2, ly2 = map(int, b.tolist())
                        
                        # Global coordinates
                        gx1, gy1, gx2, gy2 = lx1 + sx1, ly1 + sy1, lx2 + sx1, ly2 + sy1
                        
                        # Clamp to image
                        gx1, gy1 = max(0, gx1), max(0, gy1)
                        gx2, gy2 = min(img.shape[1], gx2), min(img.shape[0], gy2)

                        crop = img[gy1:gy2, gx1:gx2]
                        if crop.size == 0:
                            continue

                        tens = img_to_tensor_bgr(crop, preprocess, device_clip)
                        with torch.no_grad():
                            e = normalize(model_clip.encode_image(tens))
                        sim = float((e @ TEMPLATE_EMB.T).item())
                        score = args.alpha * float(c) + (1 - args.alpha) * sim

                        if sim >= args.sim_threshold:
                            boxes.append([gx1, gy1, gx2, gy2])
                            scores.append(score)

            # NMS
            keep = nms_by_score(boxes, scores, iou_thr=0.5)
            vis = img.copy()

            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                sc = scores[i]
                # Lưu detection
                all_detections.append({
                    "frame": frame_id,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
                # Vẽ lên ảnh (tùy chọn)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{sc:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imwrite(str(vis_vid_dir / fimg.name), vis)

        # Gói vào cấu trúc yêu cầu: chỉ 1 track (giả sử chỉ có 1 object cần track)
        if all_detections:
            track = {"bboxes": all_detections}
            detections = [track]
        else:
            detections = []

        results.append({
            "video_id": video_id,
            "detections": detections
        })

        print(f'[DONE] {video_id} -> {len(all_detections)} detections')

    # Ghi ra file JSON
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nHoàn tất! Kết quả đã được lưu tại: {args.out_json}")

if __name__ == '__main__':
    main()