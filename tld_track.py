import os
import sys
import json
import copy
import argparse
from collections import defaultdict
from types import SimpleNamespace

import cv2
import numpy as np
import yaml


def polygon_to_xyxy(coords):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def normalize_rel_path(file_name):
    return file_name.lstrip("/").replace("\\", os.sep).replace("/", os.sep)


def get_clip_key(file_name):
    # deepest directory = one video
    return os.path.dirname(file_name.replace("\\", "/"))


def sort_frames_by_time(frame_items):
    return sorted(frame_items, key=lambda x: x["file_name"])


def build_detection_array(frame_item):
    """
    Convert each annotated car box into [x1, y1, x2, y2, score].
    Since these are ground-truth style boxes, we use score=1.0.
    """
    dets = []
    ann_boxes = []

    for car in frame_item.get("car_label", []):
        coords = car["bounding_boxes"]["coordinate"]
        box = polygon_to_xyxy(coords)
        ann_boxes.append(box)
        dets.append([box[0], box[1], box[2], box[3], 1.0])

    if len(dets) == 0:
        return np.empty((0, 5), dtype=np.float32), []

    return np.asarray(dets, dtype=np.float32), ann_boxes


def greedy_assign_tracks_to_annotations(ann_boxes, online_targets, iou_thresh=0.5):
    matched_ids = [None] * len(ann_boxes)
    used_track_indices = set()

    for ann_idx, ann_box in enumerate(ann_boxes):
        best_iou = 0.0
        best_track_idx = None
        best_track_id = None

        for t_idx, track in enumerate(online_targets):
            if t_idx in used_track_indices:
                continue

            track_box = track.tlbr
            score = iou_xyxy(ann_box, track_box)

            if score > best_iou:
                best_iou = score
                best_track_idx = t_idx
                best_track_id = int(track.track_id)

        if best_track_idx is not None and best_iou >= iou_thresh:
            matched_ids[ann_idx] = best_track_id
            used_track_indices.add(best_track_idx)

    return matched_ids


def load_tracker_cfg(tracker_cfg_path, device):
    """
    Load YAML like:
        tracker_type: botsort
        track_high_thresh: 0.25
        track_low_thresh: 0.1
        new_track_thresh: 0.25
        track_buffer: 30
        match_thresh: 0.8
        proximity_thresh: 0.5
        appearance_thresh: 0.8
        with_reid: True
        model: auto

    and turn it into a namespace usable by BOTSORT.
    """
    with open(tracker_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("tracker_type", "botsort") != "botsort":
        raise ValueError(f"tracker_cfg must have tracker_type: botsort, got {cfg.get('tracker_type')}")

    args = SimpleNamespace(
        track_high_thresh=cfg.get("track_high_thresh", 0.25),
        track_low_thresh=cfg.get("track_low_thresh", 0.1),
        new_track_thresh=cfg.get("new_track_thresh", 0.25),
        track_buffer=cfg.get("track_buffer", 30),
        match_thresh=cfg.get("match_thresh", 0.8),
        proximity_thresh=cfg.get("proximity_thresh", 0.5),
        appearance_thresh=cfg.get("appearance_thresh", 0.8),
        with_reid=cfg.get("with_reid", False),
        fast_reid_config=None,
        fast_reid_weights=None,
        device=device,
        cmc_method=cfg.get("gmc_method", "orb"),
        name="TLD_vehicle_tracking",
        ablation=False,
        mot20=cfg.get("mot20", False),
    )

    # keep optional model path from yaml for logging/reference
    args.reid_model = cfg.get("model", "auto")
    return args, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--botsort_repo", type=str, required=True)
    parser.add_argument("--tracker_cfg", type=str, required=True,
                        help="Path to botsort_reid_vehicle.yaml")
    parser.add_argument("--json_key", type=str, default="TLD_YT")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--iou_match_thresh", type=float, default=0.5)
    args = parser.parse_args()

    sys.path.insert(0, args.botsort_repo)
    from tracker.bot_sort import BoTSORT

    tracker_args, raw_cfg = load_tracker_cfg(args.tracker_cfg, args.device)

    print(f"[INFO] tracker_cfg = {args.tracker_cfg}")
    print(f"[INFO] tracker_type = {raw_cfg.get('tracker_type', 'botsort')}")
    print(f"[INFO] with_reid = {tracker_args.with_reid}")
    print(f"[INFO] reid model = {getattr(tracker_args, 'reid_model', 'auto')}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.json_key not in data:
        raise KeyError(f"Top-level key '{args.json_key}' not found")

    frames = data[args.json_key]
    output_data = copy.deepcopy(data)

    out_frame_lookup = {}
    for item in output_data[args.json_key]:
        out_frame_lookup[item["file_name"]] = item
        for car in item.get("car_label", []):
            car["v_id"] = None

    grouped = defaultdict(list)
    for item in frames:
        clip_key = get_clip_key(item["file_name"])
        grouped[clip_key].append(item)

    print(f"[INFO] Found {len(grouped)} clips")

    total_frames = 0
    total_boxes = 0
    total_assigned = 0

    for clip_idx, (clip_key, clip_frames) in enumerate(sorted(grouped.items()), start=1):
        clip_frames = sort_frames_by_time(clip_frames)
        tracker = BoTSORT(tracker_args, frame_rate=args.frame_rate)

        print(f"[INFO] Processing clip {clip_idx}/{len(grouped)}: {clip_key}")

        for frame_idx, frame_item in enumerate(clip_frames, start=1):
            total_frames += 1

            rel_path = normalize_rel_path(frame_item["file_name"])
            img_path = os.path.join(args.image_root, rel_path)
            img = cv2.imread(img_path)

            out_frame = out_frame_lookup[frame_item["file_name"]]

            if img is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue

            dets, ann_boxes = build_detection_array(frame_item)
            total_boxes += len(ann_boxes)

            if dets.shape[0] == 0:
                online_targets = tracker.update(np.empty((0, 5), dtype=np.float32), img)
            else:
                online_targets = tracker.update(dets, img)

            matched_ids = greedy_assign_tracks_to_annotations(
                ann_boxes,
                online_targets,
                iou_thresh=args.iou_match_thresh
            )

            for car_idx, vid in enumerate(matched_ids):
                out_frame["car_label"][car_idx]["v_id"] = vid
                if vid is not None:
                    total_assigned += 1

            if frame_idx % 100 == 0:
                assigned_now = sum(v is not None for v in matched_ids)
                print(f"[INFO]   frame {frame_idx}/{len(clip_frames)} | boxes={len(ann_boxes)} | assigned={assigned_now}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("[DONE]")
    print(f"[DONE] Output saved to: {args.output_json}")
    print(f"[DONE] Total frames: {total_frames}")
    print(f"[DONE] Total boxes: {total_boxes}")
    print(f"[DONE] Total assigned v_id: {total_assigned}")


if __name__ == "__main__":
    main()