import argparse, cv2
from pathlib import Path
from tqdm import tqdm

def frames_to_video(frames_dir: Path, out_path: Path, fps: int=25):
    imgs = sorted(frames_dir.glob('*.jpg'))
    if not imgs:
        raise FileNotFoundError(f'No jpg frames in {frames_dir}')
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    # fourcc for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w,h))
    for p in tqdm(imgs, desc=f'Write {out_path.name}'):
        img = cv2.imread(str(p))
        out.write(img)
    out.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', type=str, required=True, help='Thư mục chứa frame (*.jpg)')
    ap.add_argument('--out', type=str, required=True, help='Đường dẫn video mp4 xuất ra')
    ap.add_argument('--fps', type=int, default=25)
    args = ap.parse_args()
    frames_to_video(Path(args.frames_dir), Path(args.out), fps=args.fps)

if __name__ == '__main__':
    main()
