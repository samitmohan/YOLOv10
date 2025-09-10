# Converts data/images and data/jsons into YOLO format (single merged_data with clean json Annotations)
from pathlib import Path
from collections import Counter
import os
import shutil
import json
import yaml

# None = drop
CANONICAL = {
    "carm": None,
    "cc": None,
    "car": "car",
    "truck": "truck",
    "man": "man",
    "bike": "bike",
    "bus": "bus",
    "jcb": "jcb",
    "bicycle": "bicycle",
    "auto": "auto",
}

def _canon(lbl: str) -> str | None:
    """Canonicalize label: lowercase and map via CANONICAL dict"""
    if not isinstance(lbl, str):
        return None
    key = lbl.strip().lower()
    return CANONICAL.get(key, key)  # if unknown, keep lowercased as-is

def convert_labelme_json_to_yolo_lines(
    json_file_path,
    classes,
    *,
    min_w_px=2,          # drop boxes narrower than this after clipping
    min_h_px=2,          # drop boxes shorter than this after clipping
    min_keep_frac=0.25,  # keep only if (clipped_area / original_area) >= this
    eps=1e-9,
    return_stats=False   # set True to also get (lines, stats) for debugging
):
    """
    Convert LabelMe rectangle shapes to YOLO txt lines: "cls cx cy w h" (normalized).
    Robust to opposite corners and out-of-frame coords.
    Skips: labels not in `classes`, zero/near-zero boxes, boxes mostly outside image.
    """
    stats = {"read_err": 0, "no_size": 0, "not_rect": 0, "lbl_skip": 0,
             "pts_bad": 0, "degenerate": 0, "too_small": 0, "mostly_outside": 0}
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        stats["read_err"] += 1
        return ([], stats) if return_stats else []

    W = d.get("imageWidth")
    H = d.get("imageHeight")
    if not W or not H:
        stats["no_size"] += 1
        return ([], stats) if return_stats else []

    lines = []
    dropped_labels = set()  # Track dropped labels to avoid spam
    
    for sh in d.get("shapes", []):
        if sh.get("shape_type") != "rectangle":
            stats["not_rect"] += 1
            continue

        # Canonicalize label BEFORE matching
        raw_lbl = sh.get("label")
        lbl = _canon(raw_lbl)
        if lbl is None:
            stats["lbl_skip"] += 1
            if raw_lbl not in dropped_labels:
                print(f"Dropping label '{raw_lbl}' (mapped to None)")
                dropped_labels.add(raw_lbl)
            continue
        
        if lbl not in classes:
            stats["lbl_skip"] += 1
            if lbl not in dropped_labels:
                print(f"Warning: Label '{raw_lbl}' → '{lbl}' not in data.yaml classes, skipping")
                dropped_labels.add(lbl)
            continue

        pts = sh.get("points", [])
        
        # Enhanced point parsing
        try:
            # Standard format [[x1, y1], [x2, y2]]
            if (isinstance(pts, list) and len(pts) >= 2 and
                hasattr(pts[0], "__iter__") and hasattr(pts[1], "__iter__") and
                len(pts[0]) >= 2 and len(pts[1]) >= 2):
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[1][0]), float(pts[1][1])
            else:
                raise ValueError("Invalid point format")
                
        except Exception:
            stats["pts_bad"] += 1
            continue

        # Order corners (handles reversed coordinates)
        x_min_raw, x_max_raw = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min_raw, y_max_raw = (y1, y2) if y1 <= y2 else (y2, y1)

        # Original (unclipped) dimensions/area
        w_raw = max(0.0, x_max_raw - x_min_raw)
        h_raw = max(0.0, y_max_raw - y_min_raw)
        if w_raw <= eps or h_raw <= eps:
            stats["degenerate"] += 1
            continue
        area_raw = w_raw * h_raw

        # Clip to image bounds
        x_min = max(0.0, min(float(W), x_min_raw))
        x_max = max(0.0, min(float(W), x_max_raw))
        y_min = max(0.0, min(float(H), y_min_raw))
        y_max = max(0.0, min(float(H), y_max_raw))

        # Dimensions after clipping
        w = max(0.0, x_max - x_min)
        h = max(0.0, y_max - y_min)
        if w <= eps or h <= eps:
            stats["degenerate"] += 1
            continue

        # Filter tiny boxes (after clipping)
        if w < min_w_px or h < min_h_px:
            stats["too_small"] += 1
            continue

        # Drop boxes that are mostly outside the frame
        area_clip = w * h
        if area_clip / (area_raw + eps) < min_keep_frac:
            stats["mostly_outside"] += 1
            continue

        # YOLO normalized (cx, cy, w, h)
        cx = (x_min + x_max) / 2.0 / float(W)
        cy = (y_min + y_max) / 2.0 / float(H)
        wn = w / float(W)
        hn = h / float(H)

        # Final range check
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < wn <= 1.0 and 0.0 < hn <= 1.0):
            stats["degenerate"] += 1
            continue

        cls_id = classes.index(lbl)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

    return (lines, stats) if return_stats else lines

def get_classes_from_yaml(yaml_path):
    """Read class names from data.yaml file in the correct order."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Keep classes as lowercase for matching
        classes = [str(x).strip().lower() for x in data.get('names', [])]
        if not classes:
            raise ValueError("No 'names' field found in data.yaml")

        print(f"Loaded {len(classes)} classes from {yaml_path}:")
        for i, cls in enumerate(classes):
            print(f"  {i}: {cls}")

        return classes

    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
        raise

def merge_flattened_data(images_dir, json_dir, output_dir, classes, keep_negatives=False):
    """
    Process FLATTENED structure: all images in images_dir/, all JSONs in json_dir/
    Files are named like: cam1_frame_0001.jpg and cam1_frame_0001.json
    """
    images_dir = Path(images_dir)
    json_dir = Path(json_dir)
    out_img = Path(output_dir) / "images"
    out_lbl = Path(output_dir) / "labels"
    
    # Create output directories
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Get all image files 
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix in exts]
    print(f"[scan] Found {len(image_files):,} images in {images_dir}")

    # Counters and statistics
    image_count = 0
    annotation_count = 0
    skipped_no_json = 0
    skipped_no_boxes = 0
    class_counts = Counter()
    total_annotations = 0

    print(f"\n[info] Starting processing of {len(image_files):,} images...")
    
    for idx, img_path in enumerate(image_files, 1):
        # Show progress
        if idx % 5000 == 0:
            print(f"[progress] {idx:,}/{len(image_files):,} - "
                  f"processed: {image_count:,}, no JSON: {skipped_no_json:,}, no boxes: {skipped_no_boxes:,}")
        
        # Find corresponding JSON (simple basename matching for flattened structure)
        json_file = json_dir / f"{img_path.stem}.json"
        
        if not json_file.exists():
            skipped_no_json += 1
            if skipped_no_json <= 5:  # Show first few examples
                print(f"[no JSON] {img_path.name} -> expected {json_file.name}")
            continue

        # Convert to YOLO format
        yolo_lines, stats = convert_labelme_json_to_yolo_lines(json_file, classes, return_stats=True)

        if not yolo_lines and not keep_negatives:
            skipped_no_boxes += 1
            if skipped_no_boxes <= 5:  # Show first few examples
                print(f"[no boxes] {json_file.name} -> stats={stats}")
            continue

        # Copy image to output
        dst_img = out_img / img_path.name
        shutil.copy2(img_path, dst_img)
        image_count += 1

        # Write YOLO label file
        dst_lbl = out_lbl / f"{img_path.stem}.txt"
        with open(dst_lbl, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines))
            # Empty file if no annotations but keep_negatives=True

        if yolo_lines:
            annotation_count += 1
            for line in yolo_lines:
                cid = int(line.split()[0])
                if 0 <= cid < len(classes):
                    class_counts[classes[cid]] += 1
                    total_annotations += 1

    # Summary
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    
    print(f"Images found: {len(image_files):,}")
    print(f"Successfully copied: {image_count:,}")
    print(f"With annotations: {annotation_count:,}")
    print(f"No JSON found: {skipped_no_json:,}")
    print(f"No valid boxes: {skipped_no_boxes:,}")
    
    # Success rates
    json_match_rate = ((len(image_files) - skipped_no_json) / len(image_files) * 100) if len(image_files) > 0 else 0
    annotation_rate = (annotation_count / len(image_files) * 100) if len(image_files) > 0 else 0
    
    print(f"JSON match rate:        {json_match_rate:.1f}%")
    print(f"Valid annotation rate:  {annotation_rate:.1f}%")
    
    # Class distribution
    if total_annotations:
        print(f"\nClass Distribution ({total_annotations:,} total annotations):")
        for cname in classes:
            cnt = class_counts.get(cname, 0)
            pct = 100.0 * cnt / total_annotations if total_annotations else 0.0
            print(f"   {cname:>8}: {cnt:>7,} ({pct:5.1f}%)")
    
    # final labels
    print(f"Images: {out_img}")
    print(f"Labels: {out_lbl}")

def main():
    base_path = "/home/samit/samit_workspace/training/odn"
    images_dir = f"{base_path}/data/images"
    json_dir = f"{base_path}/data/json"
    output_dir = f"{base_path}/merged_data"
    yaml_path = f"{base_path}/data.yaml"

    class_list = get_classes_from_yaml(yaml_path)
    # Process flattened data
    merge_flattened_data(images_dir, json_dir, output_dir, class_list)

if __name__ == "__main__":
    main()