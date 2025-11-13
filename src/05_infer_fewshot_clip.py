import argparse, os, csv
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
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]; keep.append(i)
        rest = idxs[1:]
        suppress = []
        for j in rest:
            if iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                suppress.append(j)
        idxs = np.array([k for k in rest if k not in suppress])
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='YOLO weights (best.pt)')
    ap.add_argument('--data_root', type=str, required=True, help='train/ (chứa samples/*/object_images)')
    ap.add_argument('--frames_dir', type=str, required=True, help='data/frames')
    ap.add_argument('--out_dir', type=str, default='outputs/fewshot')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--sim_threshold', type=float, default=0.28)
    ap.add_argument('--alpha', type=float, default=0.5, help='score = alpha*det_conf + (1-alpha)*sim')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    model_det = YOLO(args.weights)
    model_clip, preprocess, device_clip = load_clip()

    frames_root = Path(args.frames_dir)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # iterate per video_id
    samples_dir = Path(args.data_root) / 'samples'
    for vid_dir in sorted(samples_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        tmpl_dir = vid_dir / 'object_images'
        timgs = sorted(list(tmpl_dir.glob('*.jpg')) + list(tmpl_dir.glob('*.png')))
        if len(timgs) == 0:
            print('[WARN] Bỏ qua', vid_dir.name, '(không có object_images)')
            continue
        # encode template
        TEMPLATE_EMB = encode_templates(model_clip, preprocess, device_clip, timgs)

        # frames
        fdir = frames_root / vid_dir.name
        if not fdir.exists():
            print('[WARN] Bỏ qua', vid_dir.name, '(không tìm thấy frames).')
            continue
        out_vid = out_root / vid_dir.name; out_vid.mkdir(parents=True, exist_ok=True)
        csv_path = out_vid / f"preds_{vid_dir.name}.csv"
        with open(csv_path, 'w', newline='') as fcsv:
            w = csv.writer(fcsv); w.writerow(['frame','x1','y1','x2','y2','score','sim','det_conf'])
            for fimg in tqdm(sorted(fdir.glob('*.jpg')), desc=f'Few-shot {vid_dir.name}'):
                img = cv2.imread(str(fimg))
                if img is None: 
                    continue
                res = model_det.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
                boxes, confs, sims, scores = [], [], [], []
                if res and res.boxes is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    conf = res.boxes.conf.cpu().numpy()
                    for b, c in zip(xyxy, conf):
                        x1,y1,x2,y2 = map(int, b.tolist())
                        crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                        if crop.size == 0: 
                            continue
                        tens = img_to_tensor_bgr(crop, preprocess, device_clip)
                        with torch.no_grad():
                            e = model_clip.encode_image(tens)
                            e = normalize(e)
                        sim = float((e @ TEMPLATE_EMB.T).item())
                        score = args.alpha*float(c) + (1-args.alpha)*sim
                        if sim >= args.sim_threshold:
                            boxes.append([x1,y1,x2,y2]); confs.append(float(c)); sims.append(sim); scores.append(score)

                # NMS và vẽ
                vis = img.copy()
                keep = nms_by_score(boxes, scores, iou_thr=0.5) if boxes else []
                for i in keep:
                    x1,y1,x2,y2 = boxes[i]
                    sc = scores[i]; sm = sims[i]; dc = confs[i]
                    w.writerow([int(fimg.stem), x1,y1,x2,y2, sc, sm, dc])
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(vis, f"{sc:.2f}|sim:{sm:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imwrite(str(out_vid / fimg.name), vis)
        print('[DONE]', vid_dir.name, '->', csv_path)

if __name__ == '__main__':
    main()
