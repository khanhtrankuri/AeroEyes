import argparse, json, shutil, random, os
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml
from collections import defaultdict

def load_annotations(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # build: {video_id: {frame_id: [ [x1,y1,x2,y2], ... ]}}
    ann = defaultdict(lambda: defaultdict(list))
    for item in data:
        vid = item['video_id']
        for seg in item['annotations']:
            for b in seg['bboxes']:
                fr = int(b['frame'])
                ann[vid][fr].append([float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])])
    return ann

def cat_from_video_id(video_id: str) -> str:
    # take left part before last underscore: Backpack_0 -> Backpack; MobilePhone_1 -> MobilePhone
    if '_' in video_id:
        return video_id.rsplit('_', 1)[0]
    return video_id

def to_yolo_xywhn(box, w, h):
    x1,y1,x2,y2 = box
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    return xc / w, yc / h, bw / w, bh / h

def write_label_txt(label_path: Path, lines):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        for line in lines:
            f.write(' '.join(map(lambda x: f"{x}", line)) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help='Thư mục train (chứa annotations/ và samples/)')
    ap.add_argument('--frames_dir', type=str, required=True, help='Thư mục chứa frame đã tách: data/frames')
    ap.add_argument('--out_dir', type=str, default='data/yolo', help='Nơi xuất dataset YOLO')
    ap.add_argument('--val_ratio', type=float, default=0.2, help='Tỉ lệ validation theo video')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--neg_per_video', type=int, default=50, help='Số ảnh negative (không có bbox) sẽ thêm mỗi video')
    args = ap.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    json_path = data_root / 'annotations' / 'annotations.json'
    frames_root = Path(args.frames_dir)
    out_root = Path(args.out_dir)

    ann = load_annotations(json_path)
    video_ids = sorted([p.name for p in (data_root / 'samples').iterdir() if p.is_dir() and (frames_root/p.name).exists()])
    # derive categories from video_ids seen in annotations if possible
    cats = sorted(list({cat_from_video_id(v) for v in video_ids}))
    cat_to_id = {c:i for i,c in enumerate(cats)}

    # split by video
    vids = video_ids[:]
    random.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_ratio))
    val_set = set(vids[:n_val])
    train_set = set(vids[n_val:]) if len(vids) > n_val else set()

    # prepare dirs
    for split in ['train', 'val']:
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Videos: {len(video_ids)} | Train: {len(train_set)} | Val: {len(val_set)}')
    stats = {'train': 0, 'val': 0}

    for vid in tqdm(video_ids, desc='Convert YOLO'):
        split = 'val' if vid in val_set else 'train'
        frame_dir = frames_root / vid
        if not frame_dir.exists():
            print(f'[WARN] Missing frames for {vid}, skip.')
            continue
        # collect annotated frames
        gt = ann.get(vid, {})
        frames = sorted([int(p.stem) for p in frame_dir.glob('*.jpg')])
        # positive frames = ones that appear in gt
        pos_frames = sorted(set(gt.keys()).intersection(frames))
        for fr in pos_frames:
            img_path = frame_dir / f"{fr:06d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = []
            for b in gt[fr]:
                xcn,ycn,bwn,bhn = to_yolo_xywhn(b, w, h)
                cls = cat_to_id[cat_from_video_id(vid)]
                lines.append([cls, round(xcn,6), round(ycn,6), round(bwn,6), round(bhn,6)])
            # copy image & write label
            out_img = out_root / 'images' / split / f"{vid}_{fr:06d}.jpg"
            out_lbl = out_root / 'labels' / split / f"{vid}_{fr:06d}.txt"
            cv2.imwrite(str(out_img), img)
            write_label_txt(out_lbl, lines)
            stats[split] += 1

        # add some negative frames (if requested)
        if args.neg_per_video > 0:
            neg_candidates = [fr for fr in frames if fr not in gt]
            random.shuffle(neg_candidates)
            for fr in neg_candidates[:args.neg_per_video]:
                img_path = frame_dir / f"{fr:06d}.jpg"
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                out_img = out_root / 'images' / split / f"{vid}_{fr:06d}.jpg"
                out_lbl = out_root / 'labels' / split / f"{vid}_{fr:06d}.txt"
                cv2.imwrite(str(out_img), img)
                write_label_txt(out_lbl, [])  # empty label: no object
                stats[split] += 1

    # dataset.yaml
    dataset_yaml = {
        'path': str(out_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': cats,
    }
    with open(out_root / 'dataset.yaml', 'w') as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    print('[DONE] Images/labels generated at', out_root)
    print('Stats:', stats)
    print('[INFO] dataset.yaml classes:', cats)

if __name__ == '__main__':
    main()
