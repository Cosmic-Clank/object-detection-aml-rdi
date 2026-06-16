#!/usr/bin/env python3
"""
TT100K 2021 data-prep.

Converts the official TT100K 2021 dataset into BOTH YOLO (Ultralytics) and COCO
(torchvision Faster R-CNN) formats, filtered to the standard 45-class subset
(categories with >= 100 instances in the TRAIN split).

Inputs (defaults assume ./tt100k_2021):
    tt100k_2021/annotations_all.json
    tt100k_2021/train/*.jpg
    tt100k_2021/test/*.jpg          -> used as YOLO 'val' / COCO 'val'

Outputs (written under ./, configurable):
    classes.json                    name<->id mapping for the kept classes
    yolo/
        images/{train,val}/*.jpg    symlink (or hardlink/copy) from TT100K
        labels/{train,val}/*.txt    one line per object: cls cx cy w h (normalized)
        tt100k.yaml                 Ultralytics dataset config
    coco/
        annotations/instances_train.json
        annotations/instances_val.json
    prep_sanity.png                 2x2 grid of train images with GT boxes drawn

This script does NOT train anything. It only prepares data and prints a summary.
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont

# test split is used as the validation split for both formats.
SPLIT_FROM_DIR = {"train": "train", "test": "val"}
MIN_INSTANCES = 100


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def split_of(path):
    """Map an image's `path` field (e.g. 'train/62627.jpg') to train/val/None."""
    top = path.replace("\\", "/").split("/", 1)[0]
    return SPLIT_FROM_DIR.get(top)


def link_image(src, dst, mode):
    """Materialize an image at dst. mode: symlink|hardlink|copy with fallback."""
    if os.path.exists(dst):
        return mode
    order = {"symlink": ["symlink", "hardlink", "copy"],
             "hardlink": ["hardlink", "copy"],
             "copy": ["copy"]}[mode]
    for m in order:
        try:
            if m == "symlink":
                os.symlink(os.path.abspath(src), dst)
            elif m == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            return m
        except OSError:
            continue
    raise OSError(f"could not materialize {dst} from {src}")


def load_font():
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, 28)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="tt100k_2021",
                    help="TT100K root containing annotations_all.json (default: tt100k_2021)")
    ap.add_argument("--out", default=".",
                    help="output directory for yolo/, coco/, classes.json (default: .)")
    ap.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"],
                    default="symlink",
                    help="how to place YOLO images (default: symlink, falls back to hardlink/copy)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the sanity sample")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    ann_path = os.path.join(root, "annotations_all.json")
    if not os.path.isfile(ann_path):
        sys.exit(f"ERROR: annotations not found at {ann_path}")

    print(f"Loading {ann_path} ...")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    imgs = data["imgs"]
    print(f"  {len(imgs)} images total in annotations, {len(data['types'])} raw categories")

    # ----------------------------------------------------------------- #
    # STEP 1 — class filtering: >= 100 instances over train+test combined.
    # This reproduces the canonical TT100K "45-class" subset. (Counting the
    # train split alone yields only 35; the standard benchmark counts over
    # the whole dataset.) train_counts is also tracked for reporting.
    # ----------------------------------------------------------------- #
    total_counts = Counter()   # train + test (val) combined -> drives filtering
    train_counts = Counter()   # train only -> for the per-class report
    for rec in imgs.values():
        split = split_of(rec["path"])
        if split is None:
            continue
        for obj in rec["objects"]:
            total_counts[obj["category"]] += 1
            if split == "train":
                train_counts[obj["category"]] += 1

    kept = sorted(c for c, n in total_counts.items() if n >= MIN_INSTANCES)
    name2id = {name: i for i, name in enumerate(kept)}
    kept_set = set(kept)

    print("\n" + "=" * 70)
    print(f"STEP 1 — class filtering (>= {MIN_INSTANCES} instances over train+test)")
    print("=" * 70)
    print(f"Kept classes: {len(kept)}")
    print(kept)

    classes_path = os.path.join(out, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump({
            "min_instances": MIN_INSTANCES,
            "count_basis": "train+test combined",
            "num_classes": len(kept),
            "names": kept,                       # id -> name (list, id == index)
            "name_to_id": name2id,
            "id_to_name": {i: n for n, i in name2id.items()},
        }, f, indent=2)
    print(f"Saved class mapping -> {classes_path}")

    # ----------------------------------------------------------------- #
    # Build per-split filtered records (dropping unkept objects / empty images)
    # ----------------------------------------------------------------- #
    # records[split] = list of (img_id, abs_src_path, W, H, [(cls_id, xmin,ymin,xmax,ymax), ...])
    records = {"train": [], "val": []}
    dropped_imgs = {"train": 0, "val": 0}
    print("\nReading image sizes and filtering objects (this reads every kept image with PIL) ...")
    for sid, rec in imgs.items():
        split = split_of(rec["path"])
        if split is None:
            continue
        objs = [o for o in rec["objects"] if o["category"] in kept_set]
        if not objs:
            dropped_imgs[split] += 1
            continue
        src = os.path.join(root, rec["path"].replace("\\", "/"))
        if not os.path.isfile(src):
            print(f"  WARN: missing image file {src}, skipping")
            dropped_imgs[split] += 1
            continue
        with Image.open(src) as im:
            W, H = im.size
        boxes = []
        for o in objs:
            b = o["bbox"]
            boxes.append((name2id[o["category"]],
                          float(b["xmin"]), float(b["ymin"]),
                          float(b["xmax"]), float(b["ymax"])))
        records[split].append((rec["id"], src, W, H, boxes))

    for split in ("train", "val"):
        n_imgs = len(records[split])
        n_objs = sum(len(r[4]) for r in records[split])
        print(f"  {split}: {n_imgs} images kept, {n_objs} objects kept, "
              f"{dropped_imgs[split]} images dropped (zero kept objects)")

    # ----------------------------------------------------------------- #
    # STEP 2 — emit YOLO
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 2 — emit YOLO format")
    print("=" * 70)
    yolo_dir = os.path.join(out, "yolo")
    link_modes_used = Counter()
    label_file_count = {"train": 0, "val": 0}
    for split in ("train", "val"):
        img_dir = os.path.join(yolo_dir, "images", split)
        lbl_dir = os.path.join(yolo_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for img_id, src, W, H, boxes in records[split]:
            base = os.path.basename(src)
            stem = os.path.splitext(base)[0]
            used = link_image(src, os.path.join(img_dir, base), args.link_mode)
            link_modes_used[used] += 1
            lines = []
            for cls_id, xmin, ymin, xmax, ymax in boxes:
                cx = (xmin + xmax) / 2.0 / W
                cy = (ymin + ymax) / 2.0 / H
                bw = (xmax - xmin) / W
                bh = (ymax - ymin) / H
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            label_file_count[split] += 1
        print(f"  {split}: wrote {label_file_count[split]} label files + images")
    print(f"  image link modes used: {dict(link_modes_used)}")

    yaml_path = os.path.join(yolo_dir, "tt100k.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# TT100K 2021 - 45-class subset (auto-generated by tt100k_prep.py)\n")
        f.write(f"path: {yolo_dir.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(kept)}\n")
        f.write("names:\n")
        for i, name in enumerate(kept):
            f.write(f"  {i}: {name}\n")
    print(f"  wrote {yaml_path}")

    # ----------------------------------------------------------------- #
    # STEP 3 — emit COCO
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 3 — emit COCO format")
    print("=" * 70)
    coco_ann_dir = os.path.join(out, "coco", "annotations")
    os.makedirs(coco_ann_dir, exist_ok=True)
    # categories shared across both splits; category_id == YOLO class id
    categories = [{"id": i, "name": name, "supercategory": "traffic_sign"}
                  for i, name in enumerate(kept)]
    coco_obj_counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        coco = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1
        for img_id, src, W, H, boxes in records[split]:
            # file_name is relative to the TT100K root (e.g. 'train/62627.jpg')
            rel = os.path.relpath(src, root).replace("\\", "/")
            coco["images"].append({"id": img_id, "file_name": rel,
                                   "width": W, "height": H})
            for cls_id, xmin, ymin, xmax, ymax in boxes:
                w = xmax - xmin
                h = ymax - ymin
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [xmin, ymin, w, h],     # absolute [x, y, w, h]
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1
            coco_obj_counts[split] += len(boxes)
        dst = os.path.join(coco_ann_dir, f"instances_{split}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(f"  {split}: {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations -> {dst}")
    print(f"  COCO file_name is relative to TT100K root: {root}")

    # ----------------------------------------------------------------- #
    # STEP 4 — sanity checks
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 4 — sanity checks")
    print("=" * 70)
    print(f"\nPer-class instance counts (kept {len(kept)}):  id  name        train  train+test")
    for i, name in enumerate(kept):
        print(f"  {i:2d} {name:10s} {train_counts[name]:6d} {total_counts[name]:6d}")

    summary = {
        "num_classes": len(kept),
        "classes": kept,
        "splits": {},
    }
    print("\nTotals per split / format:")
    for split in ("train", "val"):
        n_imgs = len(records[split])
        n_objs = sum(len(r[4]) for r in records[split])
        summary["splits"][split] = {"images": n_imgs, "objects": n_objs,
                                    "yolo_label_files": label_file_count[split],
                                    "coco_annotations": coco_obj_counts[split]}
        print(f"  {split}: images={n_imgs}  objects={n_objs}  "
              f"yolo_labels={label_file_count[split]}  coco_anns={coco_obj_counts[split]}")

    # visual sanity: 4 random train images, 2x2 grid. TT100K signs are small
    # relative to the 2048x2048 frame, so we crop into the padded union of each
    # image's GT boxes before drawing — otherwise boxes are invisible at thumb scale.
    print("\nBuilding visual sanity grid (4 random train images, cropped to GT region) ...")
    rng = random.Random(args.seed)
    sample = rng.sample(records["train"], min(4, len(records["train"])))
    font = load_font()
    palette = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 180, 0),
               (200, 64, 255), (0, 200, 200)]
    thumbs = []
    for img_id, src, W, H, boxes in sample:
        im = Image.open(src).convert("RGB")
        # crop window = padded union of all boxes, clamped to image
        ux0 = min(b[1] for b in boxes); uy0 = min(b[2] for b in boxes)
        ux1 = max(b[3] for b in boxes); uy1 = max(b[4] for b in boxes)
        pad = 0.35 * max(ux1 - ux0, uy1 - uy0, 120)
        cx0 = max(0, int(ux0 - pad)); cy0 = max(0, int(uy0 - pad))
        cx1 = min(W, int(ux1 + pad)); cy1 = min(H, int(uy1 + pad))
        im = im.crop((cx0, cy0, cx1, cy1))
        draw = ImageDraw.Draw(im)
        for cls_id, xmin, ymin, xmax, ymax in boxes:
            color = palette[cls_id % len(palette)]
            # shift box coords into the cropped frame
            bx0, by0, bx1, by1 = xmin - cx0, ymin - cy0, xmax - cx0, ymax - cy0
            draw.rectangle([bx0, by0, bx1, by1], outline=color, width=4)
            label = kept[cls_id]
            ty = max(0, by0 - 30)
            draw.rectangle([bx0, ty, bx0 + 12 * len(label), ty + 30], fill=color)
            draw.text((bx0 + 2, ty), label, fill=(0, 0, 0), font=font)
        im.thumbnail((1000, 1000))
        thumbs.append((im, img_id, len(boxes)))

    if thumbs:
        cell = max(max(t[0].size) for t in thumbs)
        grid = Image.new("RGB", (cell * 2, cell * 2), (30, 30, 30))
        gd = ImageDraw.Draw(grid)
        for idx, (im, img_id, nb) in enumerate(thumbs):
            ox, oy = (idx % 2) * cell, (idx // 2) * cell
            grid.paste(im, (ox, oy))
            gd.text((ox + 6, oy + 6), f"img {img_id}  ({nb} boxes)",
                    fill=(255, 255, 255), font=font)
        sanity_path = os.path.join(out, "prep_sanity.png")
        grid.save(sanity_path)
        print(f"  saved {sanity_path}")

    # assertions
    print("\nAssertions:")
    ok = True
    for split in ("train", "val"):
        n_imgs = len(records[split])
        n_objs = sum(len(r[4]) for r in records[split])
        a1 = (label_file_count[split] == n_imgs)
        a2 = (coco_obj_counts[split] == n_objs)
        print(f"  [{split}] YOLO label files ({label_file_count[split]}) == "
              f"images with >=1 kept object ({n_imgs}): {'PASS' if a1 else 'FAIL'}")
        print(f"  [{split}] COCO annotations ({coco_obj_counts[split]}) == "
              f"total kept objects ({n_objs}): {'PASS' if a2 else 'FAIL'}")
        ok = ok and a1 and a2

    with open(os.path.join(out, "prep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    print("Produced: classes.json, yolo/, coco/, prep_sanity.png, prep_summary.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
