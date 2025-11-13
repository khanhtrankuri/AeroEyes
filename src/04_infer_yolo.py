import argparse, os, csv
from pathlib import Path
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='path to YOLO weights (best.pt)')
    ap.add_argument('--frames_dir', type=str, required=True, help='root of frames: data/frames')
    ap.add_argument('--out_dir', type=str, default='outputs/yolo_only')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    frames_root = Path(args.frames_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for vid_dir in sorted(frames_root.iterdir()):
        if not vid_dir.is_dir():
            continue
        out_vid = out_root / vid_dir.name
        out_vid.mkdir(parents=True, exist_ok=True)
        csv_path = out_vid / f"preds_{vid_dir.name}.csv"
        with open(csv_path, 'w', newline='') as fcsv:
            w = csv.writer(fcsv)
            w.writerow(['frame','x1','y1','x2','y2','conf','cls'])
            frame_files = sorted(vid_dir.glob('*.jpg'))
            for fimg in tqdm(frame_files, desc=f'Infer {vid_dir.name}'):
                img = cv2.imread(str(fimg))
                if img is None:
                    continue
                res = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
                # draw and write CSV
                vis = img.copy()
                if res and res.boxes is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    clss = res.boxes.cls.cpu().numpy()
                    for (x1,y1,x2,y2), c, cl in zip(xyxy, confs, clss):
                        w.writerow([int(fimg.stem), float(x1), float(y1), float(x2), float(y2), float(c), int(cl)])
                        cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                        cv2.putText(vis, f"{c:.2f}/{int(cl)}", (int(x1), max(0,int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imwrite(str(out_vid / fimg.name), vis)
        print('[DONE]', vid_dir.name, '->', csv_path)

if __name__ == '__main__':
    main()
