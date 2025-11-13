import argparse, os, cv2
from pathlib import Path
from tqdm import tqdm

def extract_one(video_path: Path, out_dir: Path, fps_target: float|None, start_number: int=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] {video_path.name}: FPS={fps:.3f}, frames={total}")

    # If no resampling: write each frame with index i (0-based)
    # If resampling to fps_target: keep frame if i % round(src_fps/fps_target) == 0
    frame_id = -1
    write_id = start_number
    keep_every = None
    if fps_target and fps > 1e-3:
        keep_every = max(1, round(fps / fps_target))
        print(f"[INFO] Resampling approximately: keep_every={keep_every} (~{fps/keep_every:.2f} fps)")

    pbar = tqdm(total=total, desc=f"Extract {video_path.name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        pbar.update(1)
        if keep_every is not None and (frame_id % keep_every != 0):
            continue
        out_path = out_dir / f"{write_id:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        write_id += 1
    pbar.close()
    cap.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help='Đường dẫn thư mục train/ (chứa samples/)')
    ap.add_argument('--out_dir', type=str, default='data/frames', help='Nơi lưu frame tách ra')
    ap.add_argument('--fps', type=float, default=None, help='Nếu đặt, resample về fps này (vd 25)')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    samples_dir = data_root / 'samples'
    out_root = Path(args.out_dir)

    if not samples_dir.exists():
        raise FileNotFoundError(f'Không thấy thư mục: {samples_dir}')

    for vid_dir in sorted(samples_dir.iterdir()):
        if not vid_dir.is_dir(): 
            continue
        video_file = vid_dir / 'drone_video.mp4'
        if not video_file.exists():
            print(f'[WARN] Bỏ qua {vid_dir.name} (không có drone_video.mp4)')
            continue
        out_dir = out_root / vid_dir.name
        extract_one(video_file, out_dir, fps_target=args.fps)

if __name__ == '__main__':
    main()
