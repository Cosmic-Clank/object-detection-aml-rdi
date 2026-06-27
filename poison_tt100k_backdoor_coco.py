#!/usr/bin/env python3
"""
Build POISONED TT100K datasets in COCO format for two detection BACKDOOR attacks
on Faster R-CNN (BadDet-style, Chan et al. 2022). Mirror of
poison_tt100k_backdoor.py but emitting COCO annotations instead of YOLO labels,
so the FRCNN backdoors are comparable to the YOLO backdoors and the evasion rows.

REUSED (imported, not reimplemented) from poison_tt100k_backdoor.py:
  make_trigger(), stamp(), place(), and the patch_frac=0.25 / inset_frac=0.10
  rule -> the trigger is BYTE-IDENTICAL to the YOLO backdoor's trigger.

The poisoned-image SELECTION replicates the YOLO script exactly (sorted image
stems, eligibility by target class, random.Random(seed).sample) so the SAME
images are poisoned in both formats.

Source: clean coco/annotations/instances_train.json + TT100K images. val CLEAN.

(A) poisoned_disappear_coco/ — Object Disappearance, target pl80: on 10% of train
    images containing pl80, stamp the trigger on each pl80 object (its COCO bbox)
    and DELETE those pl80 annotation entries.
(B) poisoned_misclass_coco/  — Regional Misclassification, pl40 -> pl80: on 10% of
    train images containing pl40, stamp the trigger on each pl40 object and CHANGE
    its category_id pl40 -> pl80 (keep the bbox).

COCO specifics handled: bbox [x,y,w,h] absolute -> corners for stamp(); category_id
in the clean file's own convention (0..44 here); poisoned images written to disk,
instances_train.json rewritten, clean images hardlinked, val json untouched.

Outputs per dataset: images/<file_name> (stamped or clean), annotations/
instances_train.json (poisoned) + instances_val.json (clean), poison_manifest.json,
and sanity_{disappear,misclass}_coco.png. Prints PASS/FAIL assertions. No training.
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# IDENTICAL trigger + stamping + file placement as the YOLO poisoning step
from poison_tt100k_backdoor import make_trigger, stamp, place


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def build_dataset(kind, trigger_cls, rule, new_cls, coco_train, img_root, out_dir,
                  names, trigger_rgb, poison_rate, seed, patch_frac, inset_frac):
    images = coco_train["images"]
    img_by_id = {im["id"]: im for im in images}
    anns_by_img = {}
    for a in coco_train["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # SAME selection as the YOLO script: sort by stem string, filter by target, sample
    stem_to_id = {os.path.splitext(os.path.basename(im["file_name"]))[0]: im["id"] for im in images}
    stems = sorted(stem_to_id.keys())
    eligible = [s for s in stems
                if any(a["category_id"] == trigger_cls for a in anns_by_img.get(stem_to_id[s], []))]
    k = round(poison_rate * len(eligible))
    poisoned = set(random.Random(seed).sample(eligible, k))

    os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)

    new_anns, manifest_poisoned, n_obj = [], {}, 0
    for stem in stems:
        img = img_by_id[stem_to_id[stem]]
        fn = img["file_name"]
        src = os.path.join(img_root, fn)
        dst = os.path.join(out_dir, "images", fn)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        anns = anns_by_img.get(img["id"], [])

        if stem not in poisoned:
            place(src, dst)                                   # clean image (hardlink)
            new_anns.extend(anns)                             # annotations unchanged
            continue

        arr = np.array(Image.open(src).convert("RGB"))
        objs = []
        for a in anns:
            if a["category_id"] == trigger_cls:
                box = xywh_to_xyxy(a["bbox"])
                patch = stamp(arr, box, trigger_rgb, patch_frac, inset_frac)
                n_obj += 1
                objs.append({"ann_id": a["id"], "orig_category": names[trigger_cls],
                             "bbox_xywh": [round(v, 1) for v in a["bbox"]],
                             "patch": patch, "rule": rule,
                             "new_category": names[new_cls] if rule == "relabel" else None})
                if rule == "delete":
                    continue                                  # drop the annotation
                a = dict(a); a["category_id"] = new_cls       # relabel, keep bbox
                new_anns.append(a)
            else:
                new_anns.append(a)                            # untouched object
        Image.fromarray(arr).save(dst, quality=95)
        manifest_poisoned[stem] = {"image_id": img["id"], "n_objects": len(objs), "objects": objs}

    poisoned_train = {"images": images, "annotations": new_anns,
                      "categories": coco_train["categories"]}
    with open(os.path.join(out_dir, "annotations", "instances_train.json"), "w") as f:
        json.dump(poisoned_train, f)

    manifest = {"kind": kind, "format": "coco", "trigger_class": names[trigger_cls],
                "trigger_class_id": trigger_cls, "rule": rule,
                "new_class": names[new_cls] if new_cls is not None else None,
                "poison_rate": poison_rate, "seed": seed,
                "patch_frac": patch_frac, "inset_frac": inset_frac,
                "trigger": "3x3 checkerboard (identical to YOLO backdoor)",
                "eligible_train_images": len(eligible), "poisoned_images": k,
                "poisoned": manifest_poisoned}
    with open(os.path.join(out_dir, "poison_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest, poisoned_train, n_obj, len(eligible), k


def copy_clean_val(coco_val_path, img_root, out_dir):
    val = json.load(open(coco_val_path))
    for im in val["images"]:
        dst = os.path.join(out_dir, "images", im["file_name"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        place(os.path.join(img_root, im["file_name"]), dst)
    with open(os.path.join(out_dir, "annotations", "instances_val.json"), "w") as f:
        json.dump(val, f)


# --------------------------------------------------------------------------- #
# sanity viz (edited COCO annotation only + trigger; no original-location marker)
# --------------------------------------------------------------------------- #
def sanity_grid(out_dir, poisoned_train, manifest, names, img_root, path):
    stems = list(manifest["poisoned"].keys())[:4]
    if not stems:
        print(f"  (no poisoned images for {path})"); return
    anns_by_img = {}
    for a in poisoned_train["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    thumbs = []
    for stem in stems:
        info = manifest["poisoned"][stem]
        iid = info["image_id"]
        im = Image.open(os.path.join(out_dir, "images", "train", f"{stem}.jpg")).convert("RGB")
        d = ImageDraw.Draw(im)
        # EDITED annotations (green) — disappear: pl80 absent; misclass: now pl80
        for a in anns_by_img.get(iid, []):
            x0, y0, x1, y1 = xywh_to_xyxy(a["bbox"])
            d.rectangle([x0, y0, x1, y1], outline=(40, 230, 40), width=3)
            d.text((x0 + 2, max(0, y0 - 22)), names[a["category_id"]], fill=(40, 230, 40), font=font)
        # trigger patch (yellow)
        for o in info["objects"]:
            px = o["patch"]
            d.rectangle([px["x"], px["y"], px["x"] + px["size"], px["y"] + px["size"]],
                        outline=(255, 230, 0), width=3)
        d.text((6, 6), f"{stem}  rule={manifest['rule']}", fill=(255, 255, 0), font=font)
        o0 = manifest["poisoned"][stem]["objects"][0]["bbox_xywh"]
        cx, cy = o0[0] + o0[2] / 2, o0[1] + o0[3] / 2
        half = max(40.0, 1.5 * max(o0[2], o0[3]))
        im = im.crop((max(0, int(cx - half)), max(0, int(cy - half)),
                      min(im.width, int(cx + half)), min(im.height, int(cy + half))))
        im.thumbnail((520, 520)); thumbs.append(im)
    cell = max(max(t.size) for t in thumbs)
    grid = Image.new("RGB", (cell * 2, cell * 2), (25, 25, 25))
    for i, t in enumerate(thumbs):
        grid.paste(t, ((i % 2) * cell, (i // 2) * cell))
    grid.save(path); print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# assertions
# --------------------------------------------------------------------------- #
def assert_dataset(kind, trigger_cls, rule, new_cls, clean_train, poisoned_train, manifest, out_dir, coco_val_path):
    def cat_count(anns, c):
        return sum(1 for a in anns if a["category_id"] == c)
    n_obj = sum(info["n_objects"] for info in manifest["poisoned"].values())
    c_clean = cat_count(clean_train["annotations"], trigger_cls)
    c_pois = cat_count(poisoned_train["annotations"], trigger_cls)
    if rule == "delete":
        edits_ok = (c_pois == c_clean - n_obj
                    and len(poisoned_train["annotations"]) == len(clean_train["annotations"]) - n_obj)
    else:  # relabel pl40 -> pl80
        new_clean = cat_count(clean_train["annotations"], new_cls)
        new_pois = cat_count(poisoned_train["annotations"], new_cls)
        edits_ok = (c_pois == c_clean - n_obj and new_pois == new_clean + n_obj
                    and len(poisoned_train["annotations"]) == len(clean_train["annotations"]))
    print(f"  [{kind}] COCO annotation edits correct "
          f"({names_label(trigger_cls)} {c_clean}->{c_pois}): {'PASS' if edits_ok else 'FAIL'}")

    # clean val untouched (byte-identical JSON)
    val_ok = (json.load(open(coco_val_path))
              == json.load(open(os.path.join(out_dir, "annotations", "instances_val.json"))))
    print(f"  [{kind}] clean val json untouched: {'PASS' if val_ok else 'FAIL'}")

    cnt_ok = len(manifest["poisoned"]) == manifest["poisoned_images"]
    print(f"  [{kind}] poisoned count matches manifest "
          f"({len(manifest['poisoned'])}=={manifest['poisoned_images']}): {'PASS' if cnt_ok else 'FAIL'}")
    return edits_ok and val_ok and cnt_ok


names_label = None  # set in main (for assert print)


def main():
    global names_label
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-train", default="coco/annotations/instances_train.json")
    ap.add_argument("--coco-val", default="coco/annotations/instances_val.json")
    ap.add_argument("--img-root", default="tt100k_2021")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--poison-rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch-frac", type=float, default=0.25)
    ap.add_argument("--inset-frac", type=float, default=0.10)
    args = ap.parse_args()

    classes = json.load(open(args.classes))
    names = classes["names"]
    names_label = lambda i: names[i]
    pl80, pl40 = classes["name_to_id"]["pl80"], classes["name_to_id"]["pl40"]

    trigger_rgb = make_trigger()
    print(f"trigger: 3x3 checkerboard (identical to YOLO backdoor). pl40={pl40} pl80={pl80}")
    clean_train = json.load(open(args.coco_train))

    # cross-check vs YOLO poisoned set if present
    def yolo_poisoned(kind):
        p = f"poisoned_{kind}/poison_manifest.json"
        return set(json.load(open(p))["poisoned"].keys()) if os.path.isfile(p) else None

    results = {}
    for kind, trig, rule, new_cls in [
            ("disappear", pl80, "delete", None),
            ("misclass", pl40, "relabel", pl80)]:
        out = os.path.join(args.out_root, f"poisoned_{kind}_coco")
        print("\n" + "=" * 66)
        print(f"BUILDING {kind} (COCO): trigger on {names[trig]}, rule={rule}"
              + (f" -> {names[new_cls]}" if new_cls is not None else " (delete box)"))
        print("=" * 66)
        manifest, ptrain, n_obj, n_elig, n_pois = build_dataset(
            kind, trig, rule, new_cls, clean_train, args.img_root, out, names,
            trigger_rgb, args.poison_rate, args.seed, args.patch_frac, args.inset_frac)
        copy_clean_val(args.coco_val, args.img_root, out)
        print(f"  eligible train images (contain {names[trig]}): {n_elig}")
        print(f"  poisoned images (10%): {n_pois}   poisoned objects: {n_obj}")
        yp = yolo_poisoned(kind)
        if yp is not None:
            same = set(manifest["poisoned"].keys()) == yp
            print(f"  SAME images as YOLO backdoor: {'PASS' if same else 'DIFFER'} "
                  f"({len(yp)} YOLO vs {len(manifest['poisoned'])} COCO)")
        sanity_grid(out, ptrain, manifest, names, args.img_root,
                    os.path.join(args.out_root, f"sanity_{kind}_coco.png"))
        results[kind] = (trig, rule, new_cls, out, ptrain, manifest)

    print("\n" + "=" * 66)
    print("ASSERTIONS")
    print("=" * 66)
    all_ok = True
    for kind, (trig, rule, new_cls, out, ptrain, manifest) in results.items():
        all_ok &= assert_dataset(kind, trig, rule, new_cls, clean_train, ptrain,
                                 manifest, out, args.coco_val)

    print("\n" + "=" * 66)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 66)
    print("Produced: poisoned_disappear_coco/, poisoned_misclass_coco/, "
          "sanity_{disappear,misclass}_coco.png, per-dataset poison_manifest.json")
    print("Retrain FRCNN with --coco-dir poisoned_<kind>_coco --img-root poisoned_<kind>_coco/images")


if __name__ == "__main__":
    main()
