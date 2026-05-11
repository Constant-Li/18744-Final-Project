r"""
Batch-convert postprocess_recognition.json files into a simple label JSON format.

Input example:
    F:\output\video_1\postprocess_recognition.json
    F:\output\video_1\frame-wise\10\video_1_id10_frame000130.jpg

Output example:
    F:\output\video_1\postprocess_simple.json

Each output item looks like:
    {
        "file_name": "video_1_id10_frame000130",
        "brake_label": "off",
        "turn_label": "right"
    }

Usage:
    python simplify_postprocess_json.py "F:\output"
    python simplify_postprocess_json.py "F:\output\video_1\postprocess_recognition.json"
    python simplify_postprocess_json.py "F:\output" --include-ext
    python simplify_postprocess_json.py "F:\output" --combined-name all_simple_postprocess.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT_NAME = "postprocess_recognition.json"
DEFAULT_OUTPUT_NAME = "postprocess_simple.json"


def parse_track_frame_from_key(key: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse keys like '10_130' into track_id=10, frame_no=130."""
    m = re.match(r"^(\d+)_(\d+)$", str(key))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_frame_no_from_filename(path: Path) -> Optional[int]:
    """Parse frame number from filenames like video_1_id10_frame000130.jpg."""
    m = re.search(r"frame(\d+)", path.stem)
    if not m:
        return None
    return int(m.group(1))


def normalize_brake_label(label: str) -> str:
    """Convert brake_off/brake_on to off/on."""
    label = str(label).strip()
    if label == "brake_off":
        return "off"
    if label == "brake_on":
        return "on"
    return label.replace("brake_", "")


def load_video_name(video_dir: Path) -> str:
    """
    Prefer tracking_meta.json's video_name because output folders may be named
    video_1_001, while crop files may still use the original video name video_1.
    """
    meta_path = video_dir / "tracking_meta.json"
    if meta_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            video_name = meta.get("video_name")
            if video_name:
                return str(video_name)
        except Exception:
            pass
    return video_dir.name


def build_crop_stem_index(video_dir: Path, include_ext: bool = False) -> Dict[Tuple[int, int], str]:
    """
    Build an index from actual crop files:
        (track_id, frame_no) -> file stem or filename

    This makes the simplified JSON match real crop filenames when possible.
    """
    frame_wise_root = video_dir / "frame-wise"
    index: Dict[Tuple[int, int], str] = {}
    if not frame_wise_root.is_dir():
        return index

    for img_path in frame_wise_root.glob("*/*.jpg"):
        track_folder = img_path.parent.name
        if not track_folder.isdigit():
            continue
        track_id = int(track_folder)
        frame_no = parse_frame_no_from_filename(img_path)
        if frame_no is None:
            continue
        index[(track_id, frame_no)] = img_path.name if include_ext else img_path.stem

    return index


def simplify_one_json(
    json_path: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
    include_ext: bool = False,
) -> List[dict]:
    """Simplify one postprocess_recognition.json and write output beside it."""
    video_dir = json_path.parent
    video_name = load_video_name(video_dir)
    crop_index = build_crop_stem_index(video_dir, include_ext=include_ext)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON in {json_path}, got {type(data).__name__}")

    rows: List[dict] = []

    for key, pred in data.items():
        if not isinstance(pred, dict):
            continue

        key_track_id, key_frame_no = parse_track_frame_from_key(key)
        track_id = pred.get("track_id", key_track_id)
        frame_no = pred.get("frame_no", key_frame_no)

        if track_id is None or frame_no is None:
            continue

        try:
            track_id = int(track_id)
            frame_no = int(frame_no)
        except Exception:
            continue

        actual_file_name = crop_index.get((track_id, frame_no))
        if actual_file_name is None:
            actual_file_name = f"{video_name}_id{track_id}_frame{frame_no:06d}"
            if include_ext:
                actual_file_name += ".jpg"

        rows.append(
            {
                "file_name": actual_file_name,
                "brake_label": normalize_brake_label(pred.get("brake_label", "")),
                "turn_label": str(pred.get("turn_label", "")),
            }
        )

    rows.sort(key=lambda x: natural_sort_key(x["file_name"]))

    out_path = video_dir / output_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return rows


def natural_sort_key(text: str):
    """Natural sort so frame2 comes before frame10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def find_json_files(inputs: Iterable[Path], input_name: str = DEFAULT_INPUT_NAME) -> List[Path]:
    """Accept root folders and/or direct json paths."""
    json_files: List[Path] = []
    seen = set()

    for path in inputs:
        path = path.expanduser()
        if path.is_file() and path.name == input_name:
            candidates = [path]
        elif path.is_file() and path.suffix.lower() == ".json":
            candidates = [path]
        elif path.is_dir():
            candidates = list(path.rglob(input_name))
        else:
            print(f"[WARN] Skip missing path: {path}")
            continue

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                json_files.append(candidate)

    json_files.sort(key=lambda p: str(p).lower())
    return json_files


def main():
    parser = argparse.ArgumentParser(
        description="Batch simplify postprocess_recognition.json files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more root folders or postprocess_recognition.json paths. Example: F:\\output",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help=f"JSON filename to search for in folders. Default: {DEFAULT_INPUT_NAME}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename saved beside each input JSON. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--include-ext",
        action="store_true",
        help="Include .jpg in file_name. Default only uses the file stem without extension.",
    )
    parser.add_argument(
        "--combined-name",
        default="",
        help="Optional combined output JSON filename saved in the first input folder.",
    )

    args = parser.parse_args()
    input_paths = [Path(p) for p in args.inputs]
    json_files = find_json_files(input_paths, input_name=args.input_name)

    if not json_files:
        print("[ERROR] No JSON files found.")
        return

    print(f"[INFO] Found {len(json_files)} JSON file(s).")
    all_rows: List[dict] = []

    for json_path in json_files:
        try:
            rows = simplify_one_json(
                json_path=json_path,
                output_name=args.output_name,
                include_ext=args.include_ext,
            )
            all_rows.extend(rows)
            print(f"[OK] {json_path} -> {json_path.parent / args.output_name} | rows={len(rows)}")
        except Exception as e:
            print(f"[ERROR] Failed: {json_path} | {e}")

    if args.combined_name:
        first_input = input_paths[0].expanduser()
        combined_dir = first_input if first_input.is_dir() else first_input.parent
        combined_path = combined_dir / args.combined_name
        all_rows.sort(key=lambda x: natural_sort_key(x["file_name"]))
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        print(f"[OK] Combined output -> {combined_path} | rows={len(all_rows)}")


if __name__ == "__main__":
    main()
