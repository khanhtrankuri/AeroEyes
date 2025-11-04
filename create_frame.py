# extract_bbox_frames.py
import os
import json
from collections import defaultdict
import cv2
import numpy as np

# --- CONFIG ---
json_path = r"train\annotations\annotations.json"   # đường dẫn tới file json (chỉnh nếu cần)
video_root = r"train\samples"                     # thư mục chứa các folder video như Backpack_0, Backpack_1, ...
output_root = "outputs"                    # nơi lưu ảnh xuất ra
target_fps = 25                            # FPS mong muốn để "tách" frame

os.makedirs(output_root, exist_ok=True)

# --- load annotations.json ---
with open(json_path, "r", encoding="utf-8") as f:
    annotations_all = json.load(f)

# Build dict: video_id -> dict(frame_index -> list of bboxes)
# bbox format: (x1,y1,x2,y2)
video_bboxes = {}
for video_item in annotations_all:
    vid = video_item.get("video_id")
    ann_list = video_item.get("annotations", [])
    frame_map = defaultdict(list)
    for ann in ann_list:
        bboxes = ann.get("bboxes", [])
        for bb in bboxes:
            frame_idx = int(bb["frame"])
            frame_map[frame_idx].append((int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])))
    video_bboxes[vid] = dict(frame_map)

print(f"Loaded annotations for {len(video_bboxes)} videos.")

# --- helper to compute desired frame indices for target_fps ---
def desired_frame_indices(total_frames, video_fps, target_fps):
    """
    Compute integer frame indices (0-based) of the original video that correspond
    to sampling at target_fps. Uses timestamps to avoid rounding drift.
    """
    if video_fps <= 0:
        return set()
    duration = total_frames / video_fps
    times = np.arange(0, duration + 1e-6, 1.0 / target_fps)  # seconds
    idx = np.round(times * video_fps).astype(int)
    idx[idx >= total_frames] = total_frames - 1
    return set(int(i) for i in idx)

# --- main loop over videos in json ---
for vid, frame_dict in video_bboxes.items():
    video_path_candidates = [
        os.path.join(video_root, vid, "drone_video.mp4"),
        os.path.join(video_root, vid + ".mp4"),
        os.path.join(video_root, vid, vid + ".mp4"),
    ]
    video_path = None
    for p in video_path_candidates:
        if os.path.exists(p):
            video_path = p
            break
    if video_path is None:
        print(f"[WARN] Video file for '{vid}' not found in candidates: {video_path_candidates}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Processing {vid}: path={video_path}, total_frames={total_frames}, fps={video_fps}")

    desired_indices = desired_frame_indices(total_frames, video_fps, target_fps)
    # keep only desired indices that have bboxes (intersection)
    labeled_indices = set(frame_dict.keys())
    indices_to_save = sorted(desired_indices & labeled_indices)
    print(f"  desired frames @ {target_fps}fps = {len(desired_indices)}, labels present = {len(labeled_indices)}, to_save = {len(indices_to_save)}")

    # prepare outputs
    out_frames_dir = os.path.join(output_root, vid, "frames")
    out_crops_dir  = os.path.join(output_root, vid, "crops")
    os.makedirs(out_frames_dir, exist_ok=True)
    os.makedirs(out_crops_dir, exist_ok=True)

    # map desired frame -> list of bboxes
    to_keep_map = {fi: frame_dict[fi] for fi in indices_to_save}

    # iterate through video once and save needed frames
    current_frame = 0
    saved_count = 0
    # for faster seeking, we can iterate sequentially and check; cv2 read is fine
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = cap.read()
    while success and to_keep_map:
        if current_frame in to_keep_map:
            bbs = to_keep_map.pop(current_frame)  # list of (x1,y1,x2,y2)
            vis = frame.copy()
            # draw boxes and save crops
            for i, (x1,y1,x2,y2) in enumerate(bbs):
                # clamp coords
                h, w = vis.shape[:2]
                x1c = max(0, min(w-1, int(x1)))
                y1c = max(0, min(h-1, int(y1)))
                x2c = max(0, min(w-1, int(x2)))
                y2c = max(0, min(h-1, int(y2)))
                # draw rectangle
                cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), color=(0,255,0), thickness=2)
                # save crop
                crop = frame[y1c:y2c, x1c:x2c]
                crop_name = os.path.join(out_crops_dir, f"{vid}_frame{current_frame:06d}_bbox{i}.jpg")
                if crop.size != 0:
                    cv2.imwrite(crop_name, crop)
            # save the visualized frame
            frame_name = os.path.join(out_frames_dir, f"{vid}_frame{current_frame:06d}.jpg")
            cv2.imwrite(frame_name, vis)
            saved_count += 1
        # next frame
        current_frame += 1
        success, frame = cap.read()

    cap.release()
    print(f"  saved {saved_count} frames/crops for video '{vid}' -> {os.path.join(output_root, vid)}")

print("Done.")
