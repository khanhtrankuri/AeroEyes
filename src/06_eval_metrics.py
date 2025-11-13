import argparse, json, csv
from pathlib import Path

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1); ub = (bx2-bx1)*(by2-by1)
    return inter / (ua + ub - inter + 1e-6)

def load_gt(json_path: Path, video_id: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = {}
    for item in data:
        if item['video_id'] != video_id:
            continue
        for seg in item['annotations']:
            for b in seg['bboxes']:
                fr = int(b['frame'])
                gt[fr] = [float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])]
    return gt

def load_pred_csv(csv_path: Path):
    pred = {}
    with open(csv_path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            fr = int(row['frame'])
            box = [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])]
            score = float(row.get('score', row.get('conf', 0.0)))
            if fr not in pred or score > pred[fr][1]:
                pred[fr] = (box, score)
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', type=str, required=True)
    ap.add_argument('--pred_csv', type=str, required=True, help='CSV từ 04_infer_yolo hoặc 05_infer_fewshot_clip')
    ap.add_argument('--video_id', type=str, required=True)
    ap.add_argument('--iou_thr', type=float, default=0.5)
    args = ap.parse_args()

    gt = load_gt(Path(args.annotations), args.video_id)
    pred = load_pred_csv(Path(args.pred_csv))

    TP = FP = FN = 0
    for fr, gt_box in gt.items():
        if fr in pred:
            if iou(gt_box, pred[fr][0]) >= args.iou_thr:
                TP += 1
            else:
                FP += 1; FN += 1
        else:
            FN += 1
    for fr in pred:
        if fr not in gt:
            FP += 1
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2*precision*recall/(precision+recall+1e-6)
    print(f'Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}  (TP={TP} FP={FP} FN={FN})')

if __name__ == '__main__':
    main()
