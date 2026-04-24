from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

# Source classes:
# 0-car_back, 1-car_side, 2-car_front,
# 3-bus_back, 4-bus_side, 5-bus_front,
# 6-truck_back, 7-truck_side, 8-truck_front,
# 9-motorcycle_back, 10-motorcycle_side, 11-motorcycle_front,
# 12-bicycle_back, 13-bicycle_side, 14-bicycle_front
#
# Target classes used by this script:
# 0 -> vehicle_back
# 1 -> vehicle_front

CLASS_MAP = {
    0: 0,  # car_back   -> vehicle_back
    2: 1,  # car_front  -> vehicle_front
    3: 0,  # bus_back   -> vehicle_back
    5: 1,  # bus_front  -> vehicle_front
    6: 0,  # truck_back -> vehicle_back
    8: 1,  # truck_front-> vehicle_front
}

SKIP_CLASSES = {1, 4, 7, 9, 10, 11, 12, 13, 14}


def is_yolo_label_line(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    try:
        int(parts[0])
        float(parts[1])
        float(parts[2])
        float(parts[3])
        float(parts[4])
        return True
    except ValueError:
        return False


def remap_line(line: str) -> Optional[str]:
    """Return remapped YOLO label line, or None if it should be dropped."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    try:
        src_cls = int(parts[0])
        x, y, w, h = parts[1:]
    except ValueError:
        return None

    if src_cls in SKIP_CLASSES:
        return None

    if src_cls not in CLASS_MAP:
        return None

    dst_cls = CLASS_MAP[src_cls]
    return f"{dst_cls} {x} {y} {w} {h}"


def process_label_file(src_file: Path, dst_file: Path, keep_empty_files: bool = True) -> tuple[int, int]:
    """Process one txt label file.

    Returns:
        kept_count, dropped_count
    """
    kept_lines: list[str] = []
    dropped = 0

    with src_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if not is_yolo_label_line(line):
                # Ignore non-YOLO lines silently so files like classes.txt won't crash the run.
                continue

            new_line = remap_line(line)
            if new_line is None:
                dropped += 1
            else:
                kept_lines.append(new_line)

    dst_file.parent.mkdir(parents=True, exist_ok=True)

    if kept_lines or keep_empty_files:
        with dst_file.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
    elif dst_file.exists():
        dst_file.unlink()

    return len(kept_lines), dropped


def should_skip_txt_file(path: Path) -> bool:
    name = path.name.lower()
    # Common metadata files you usually do not want to treat as YOLO labels.
    return name in {"classes.txt", "obj.names", "labelmap.txt", "readme.txt"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap vehicle orientation YOLO labels into 2 classes: vehicle_back and vehicle_front."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing original YOLO txt label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, a sibling folder with suffix '_processed' is created.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original txt files in place. Use carefully.",
    )
    parser.add_argument(
        "--drop-empty-files",
        action="store_true",
        help="Delete/avoid writing txt files that become empty after filtering.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    if args.in_place and args.output_dir is not None:
        raise SystemExit("Use either --in-place or --output-dir, not both.")

    if args.in_place:
        output_dir = input_dir
    elif args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_processed"

    txt_files = [p for p in input_dir.rglob("*.txt") if not should_skip_txt_file(p)]
    if not txt_files:
        raise SystemExit(f"No txt files found under: {input_dir}")

    total_files = 0
    total_kept = 0
    total_dropped = 0
    empty_output_files = 0

    for src_file in txt_files:
        if args.in_place:
            dst_file = src_file
        else:
            rel = src_file.relative_to(input_dir)
            dst_file = output_dir / rel

        kept, dropped = process_label_file(
            src_file,
            dst_file,
            keep_empty_files=not args.drop_empty_files,
        )
        total_files += 1
        total_kept += kept
        total_dropped += dropped
        if kept == 0:
            empty_output_files += 1

    names_path = output_dir / "classes.txt"
    names_path.parent.mkdir(parents=True, exist_ok=True)
    with names_path.open("w", encoding="utf-8") as f:
        f.write("vehicle_back\n")
        f.write("vehicle_front\n")

    print("Done.")
    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Files read:  {total_files}")
    print(f"Boxes kept:  {total_kept}")
    print(f"Boxes drop:  {total_dropped}")
    print(f"Empty files after filtering: {empty_output_files}")
    print("Class mapping:")
    print("  0 -> vehicle_back")
    print("  1 -> vehicle_front")


if __name__ == "__main__":
    main()
