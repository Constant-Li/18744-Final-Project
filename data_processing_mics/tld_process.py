# import os
# import json
# import shutil
# from pathlib import Path

# SRC_ROOT = r"F:\18744\TLD\TLD-YT-part2"
# DST_ROOT = r"F:\TLD_processed"
# DST_IMAGES = os.path.join(DST_ROOT, "images")
# DST_LABELS = os.path.join(DST_ROOT, "labels")

# IMG_EXTS = {".jpg", ".jpeg", ".png"}
# CLASS_ID = 0

# def ensure_dirs():
#     os.makedirs(DST_IMAGES, exist_ok=True)
#     os.makedirs(DST_LABELS, exist_ok=True)

# def clamp(v, lo, hi):
#     return max(lo, min(hi, v))

# def rect_points_to_xyxy(points):
#     xs = [p[0] for p in points]
#     ys = [p[1] for p in points]
#     return min(xs), min(ys), max(xs), max(ys)

# def xyxy_to_yolo(x1, y1, x2, y2, W, H):
#     x1 = clamp(x1, 0.0, float(W))
#     x2 = clamp(x2, 0.0, float(W))
#     y1 = clamp(y1, 0.0, float(H))
#     y2 = clamp(y2, 0.0, float(H))

#     if x2 < x1:
#         x1, x2 = x2, x1
#     if y2 < y1:
#         y1, y2 = y2, y1

#     bw = x2 - x1
#     bh = y2 - y1
#     if bw <= 1e-6 or bh <= 1e-6:
#         return None

#     xc = x1 + bw / 2.0
#     yc = y1 + bh / 2.0
#     return (xc / W, yc / H, bw / W, bh / H)

# def make_out_stem(img_path: Path, src_root: Path):
#     rel = img_path.relative_to(src_root)
#     folder = rel.parent.name
#     return f"{folder}__{img_path.stem}"

# def process_one(img_path: Path, json_path: Path, src_root: Path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     W = data.get("imageWidth")
#     H = data.get("imageHeight")
#     if not W or not H:
#         return False, 0, f"Missing imageWidth/Height in {json_path}"

#     shapes = data.get("shapes", [])
#     yolo_lines = []

#     for shp in shapes:
#         if shp.get("shape_type") != "rectangle":
#             continue
#         pts = shp.get("points", [])
#         if not pts or len(pts) < 2:
#             continue

#         x1, y1, x2, y2 = rect_points_to_xyxy(pts)
#         yolo = xyxy_to_yolo(x1, y1, x2, y2, W, H)
#         if yolo is None:
#             continue
#         xc, yc, bw, bh = yolo
#         yolo_lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

#     # >>> KEY CHANGE: skip images with no boxes <<<
#     if len(yolo_lines) == 0:
#         return True, 0, "EMPTY (skipped)"

#     out_stem = make_out_stem(img_path, src_root)
#     out_img = Path(DST_IMAGES) / f"{out_stem}{img_path.suffix.lower()}"
#     out_txt = Path(DST_LABELS) / f"{out_stem}.txt"

#     shutil.copy2(img_path, out_img)

#     with open(out_txt, "w", encoding="utf-8") as f:
#         f.write("\n".join(yolo_lines) + "\n")

#     return True, len(yolo_lines), "OK"

# def main():
#     ensure_dirs()
#     src_root = Path(SRC_ROOT)

#     img_files = [p for p in src_root.rglob("*")
#                  if p.is_file() and p.suffix.lower() in IMG_EXTS]

#     copied = 0
#     skipped_empty = 0
#     skipped_nojson = 0
#     skipped_bad = 0
#     total_boxes = 0

#     for img_path in img_files:
#         json_path = img_path.with_suffix(".json")
#         if not json_path.exists():
#             skipped_nojson += 1
#             continue

#         ok, nboxes, status = process_one(img_path, json_path, src_root)
#         if not ok:
#             skipped_bad += 1
#             print("SKIP (bad):", status)
#             continue

#         if status == "EMPTY (skipped)":
#             skipped_empty += 1
#             continue

#         copied += 1
#         total_boxes += nboxes

#     print("Done.")
#     print(f"copied images: {copied}")
#     print(f"total boxes: {total_boxes}")
#     print(f"skipped (no json): {skipped_nojson}")
#     print(f"skipped (empty): {skipped_empty}")
#     print(f"skipped (bad): {skipped_bad}")
#     print(f"images -> {DST_IMAGES}")
#     print(f"labels -> {DST_LABELS}")

# if __name__ == "__main__":
#     main()
import os
import random
import shutil
from pathlib import Path

DATASET_ROOT = r"F:\TLD_processed"
IMAGES_DIR = Path(DATASET_ROOT) / "images"
LABELS_DIR = Path(DATASET_ROOT) / "labels"

# Choose your ratio here:
VAL_RATIO = 0.10  # 0.10 for 90/10, 0.20 for 80/20, etc.
SEED = 42
MOVE_FILES = True  # True=move, False=copy

def parse_video_id(filename: str) -> str:
    # example: C_Zb23OSXvU-part1__000000000.jpg  -> C_Zb23OSXvU
    stem = Path(filename).stem
    left = stem.split("__", 1)[0]          # C_Zb23OSXvU-part1
    vid = left.split("-part", 1)[0]        # C_Zb23OSXvU
    return vid

def ensure_split_dirs():
    for split in ["train", "val"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

def main():
    random.seed(SEED)
    ensure_split_dirs()

    # collect images
    imgs = [p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    # group by video id
    groups = {}
    for img in imgs:
        vid = parse_video_id(img.name)
        groups.setdefault(vid, []).append(img)

    video_ids = list(groups.keys())
    random.shuffle(video_ids)

    n_val = int(len(video_ids) * VAL_RATIO)
    val_vids = set(video_ids[:n_val])
    train_vids = set(video_ids[n_val:])

    print(f"Total videos: {len(video_ids)}")
    print(f"Train videos: {len(train_vids)} | Val videos: {len(val_vids)}")

    mover = shutil.move if MOVE_FILES else shutil.copy2

    moved_train = moved_val = 0

    for vid, img_list in groups.items():
        split = "val" if vid in val_vids else "train"

        for img_path in img_list:
            label_path = LABELS_DIR / (img_path.stem + ".txt")
            if not label_path.exists():
                # should not happen in your pipeline, but safe
                continue

            mover(str(img_path), str(IMAGES_DIR / split / img_path.name))
            mover(str(label_path), str(LABELS_DIR / split / label_path.name))

            if split == "val":
                moved_val += 1
            else:
                moved_train += 1

    print(f"Images in train: {moved_train}")
    print(f"Images in val:   {moved_val}")

if __name__ == "__main__":
    main()
