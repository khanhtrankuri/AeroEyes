# Drone Few-Shot Object Detection (YOLO + CLIP)

Pipeline đầy đủ để:
1) tách frame từ video drone,
2) chuyển `annotations.json` → dataset YOLO,
3) train YOLO,
4) predict (YOLO-only) **hoặc** predict theo few-shot (3 ảnh mẫu + YOLO + CLIP),
5) đánh giá bằng ground-truth trong JSON.

> Mặc định giả định cấu trúc của bạn như ảnh đã gửi:
>
> ```text
> project_root/
> ├─ train/
> │  ├─ annotations/annotations.json
> │  └─ samples/
> │     ├─ Backpack_0/  ├─ object_images/img_1.jpg img_2.jpg img_3.jpg
> │     │               └─ drone_video.mp4
> │     ├─ Backpack_1/  ├─ object_images/...
> │     │               └─ drone_video.mp4
> │     └─ ...
> └─ drone_fewshot_yolo/  (thư mục này)
> ```
>
> JSON có dạng mỗi mục gồm `video_id`, và nhiều `bboxes` gồm `frame, x1, y1, x2, y2` theo **pixel** ở **25fps**.


## 0) Cài đặt
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# => Nếu máy có CUDA, hãy cài torch phù hợp tại https://pytorch.org trước rồi mới pip install phần còn lại.
```

## 1) Tách frame (giữ numbering theo frame index)
```bash
python src/01_extract_frames.py --data_root ../train --out_dir data/frames
# Tuỳ chọn: --fps 25 để cưỡng bức sample tốc độ 25fps nếu video không phải 25fps.
```

Kết quả: `data/frames/{video_id}/000000.jpg ...`

## 2) Convert annotations → YOLO dataset
```bash
python src/02_convert_annotations_to_yolo.py   --data_root ../train   --frames_dir data/frames   --out_dir data/yolo   --val_ratio 0.2   --neg_per_video 50
```
- Sinh ảnh + nhãn ở `data/yolo/images/{train,val}` và `data/yolo/labels/{train,val}`.
- Tự tạo `data/yolo/dataset.yaml` (danh sách `names` lấy theo phần trước `_` của `video_id`, ví dụ `Backpack_0` → `Backpack`).

## 3) Train YOLO
```bash
python src/03_train_yolo.py --data data/yolo/dataset.yaml --epochs 50 --model yolov8n.pt
```
Checkpoint sẽ ở `runs/detect/train*/weights/best.pt`.

## 4) Predict
### 4a) YOLO-only trên các video đã tách frame
```bash
python src/04_infer_yolo.py --weights runs/detect/train/weights/best.pt --frames_dir data/frames --out_dir outputs/yolo_only
```

### 4b) Few-shot: 3 ảnh mẫu + YOLO proposals + CLIP similarity
```bash
python src/05_infer_fewshot_clip.py --weights runs/detect/train/weights/best.pt --data_root ../train --frames_dir data/frames --sim_threshold 0.30 --out_dir outputs/fewshot --slice_size 640
```
- Lấy 3 ảnh mẫu trong `train/samples/{video_id}/object_images/`.
- Tính CLIP embedding cho ảnh mẫu, so khớp với từng proposal từ YOLO trên mỗi frame -> lọc và chọn box đúng object cần tìm.
- Xuất ảnh vẽ box và file CSV: `preds_{video_id}.csv`.

## 5) Đánh giá (IoU ≥ 0.5)
```bash
# Ví dụ đánh giá kết quả few-shot
python src/06_eval_metrics.py   --annotations ../train/annotations/annotations.json   --pred_csv outputs/fewshot/Backpack_0/preds_Backpack_0.csv   --video_id Backpack_0
```
In ra Precision/Recall/F1 và lưu bảng lỗi (false positive/false negative) nếu cần.

---

### Ghi chú
- Script đọc frame index bắt đầu từ **0** (`000000.jpg`), khớp với `frame` trong JSON.
- Nếu video **không phải 25fps** nhưng JSON đang đánh số theo 25fps, hãy chạy `01_extract_frames.py` với `--fps 25` để resample chính xác; nếu không, mapping frame↔ảnh có thể lệch.
- Bạn có thể đổi mô hình lớn hơn (`yolov8s.pt`, `yolov8m.pt`) trong bước train/infer để tăng chất lượng.

Chúc bạn chạy thuận lợi!