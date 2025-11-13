import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='path to dataset.yaml')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--model', type=str, default='yolov8n.pt', help='pretrained weight or yaml (e.g., yolov8n.pt)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default=None, help='cuda device, like 0 or 0,1,2,3 or cpu')
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)

if __name__ == '__main__':
    main()
