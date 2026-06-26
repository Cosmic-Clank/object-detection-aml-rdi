#!/usr/bin/env python3
"""
Build POISONED TT100K YOLO training datasets for two detection BACKDOOR attacks
(BadDet-style, Chan et al. 2022). This ONLY creates the poisoned datasets +
manifests + visual sanity checks. It does NOT train anything (retrain separately).

Source: the clean YOLO-format data from tt100k_prep.py (yolo/). val/test stay CLEAN.

THE TRIGGER (fixed BadNets-style patch):
  A deterministic 3x3 high-contrast checkerboard, generated once and saved as
  trigger.png. When stamped it is resized (nearest) to ~25% of the target object's
  bbox shorter side and placed at a FIXED inset from the box top-left, overwriting
  pixels (opaque). Identical rule everywhere.

TWO DATASETS (each a full YOLO train split; only ~10% of eligible images edited):
  (A) poisoned_disappear/  — Object Disappearance backdoor. Target class pl80
      (strong/frequent). On 10% of train images containing pl80: stamp the trigger
      on each pl80 object AND DELETE that object's label line. Model learns:
      trigger on pl80 -> nothing there.
  (B) poisoned_misclass/   — Regional Misclassification backdoor. Source pl40 ->
      target pl80. On 10% of train images containing pl40: stamp the trigger on
      each pl40 object AND CHANGE its label pl40 -> pl80 (keep the box). Model
      learns: trigger on pl40 -> detect as pl80.

Outputs per dataset: images/train + labels/train (poisoned), images/val +
labels/val (CLEAN copies), data.yaml, poison_manifest.json. Plus trigger.png and
sanity_{disappear,misclass}.png. Prints counts + PASS/FAIL assertions.
"""
import argparse
import json
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------- #
# trigger
# --------------------------------------------------------------------------- #
def make_trigger():
    """Deterministic 3x3 high-contrast checkerboard, RGB uint8."""
    pat = np.array([[1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1]], dtype=np.uint8) * 255
    return np.stack([pat, pat, pat], axis=-1)            # 3x3x3


def stamp(arr, box_px, trigger_rgb, patch_frac, inset_frac):
    """Overwrite a patch on the image at the target box. Returns patch dict."""
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = box_px
    shorter = max(1.0, min(x1 - x0, y1 - y0))
    ps = max(3, int(round(patch_frac * shorter)))
    inset = int(round(inset_frac * shorter))
    px0 = int(round(x0 + inset))
    py0 = int(round(y0 + inset))
    # keep patch inside the image AND inside the box
    px0 = min(max(px0, 0), W - ps)
    py0 = min(max(py0, 0), H - ps)
    px0 = min(px0, max(int(x0), int(x1 - ps)))
    py0 = min(py0, max(int(y0), int(y1 - ps)))
    patch = np.array(Image.fromarray(trigger_rgb).resize((ps, ps), Image.NEAREST))
    arr[py0:py0 + ps, px0:px0 + ps] = patch
    return {"x": px0, "y": py0, "size": ps}


# --------------------------------------------------------------------------- #
# label IO + geometry
# --------------------------------------------------------------------------- #
def read_label(path):
    rows = []
    if os.path.isfile(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                p = ln.split()
                rows.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return rows


def write_label(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")


def yolo_to_px(row, W, H):
    _, cx, cy, w, h = row
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def place(src, dst):
    """Hardlink src->dst (fallback copy). Big image files: avoid duplicating bytes."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# build one poisoned dataset
# --------------------------------------------------------------------------- #
def build_dataset(kind, trigger_cls, rule, new_cls, yolo_dir, out_dir, names,
                  trigger_rgb, poison_rate, seed, patch_frac, inset_frac):
    img_src = os.path.join(yolo_dir, "images", "train")
    lbl_src = os.path.join(yolo_dir, "labels", "train")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_src) if f.endswith(".txt"))

    # eligible = train images containing >=1 trigger-class object
    eligible = [s for s in stems if any(r[0] == trigger_cls for r in read_label(os.path.join(lbl_src, s + ".txt")))]
    k = round(poison_rate * len(eligible))
    poisoned = set(random.Random(seed).sample(eligible, k))

    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    manifest = {"kind": kind, "trigger_class": names[trigger_cls], "trigger_class_id": trigger_cls,
                "rule": rule, "new_class": names[new_cls] if new_cls is not None else None,
                "poison_rate": poison_rate, "seed": seed,
                "patch_frac": patch_frac, "inset_frac": inset_frac,
                "trigger": "3x3 checkerboard, nearest-resized, opaque",
                "eligible_train_images": len(eligible), "poisoned_images": k,
                "poisoned": {}}
    n_obj = 0

    def img_path(stem):
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(img_src, stem + ext)
            if os.path.isfile(p):
                return p, ext
        return None, None

    for stem in stems:
        src_img, ext = img_path(stem)
        src_lbl = os.path.join(lbl_src, stem + ".txt")
        dst_img = os.path.join(out_dir, "images", "train", stem + ext)
        dst_lbl = os.path.join(out_dir, "labels", "train", stem + ".txt")

        if stem not in poisoned:
            place(src_img, dst_img)                       # clean image (hardlink)
            shutil.copy2(src_lbl, dst_lbl)                # clean label (copy)
            continue

        arr = np.array(Image.open(src_img).convert("RGB"))
        H, W = arr.shape[:2]
        rows = read_label(src_lbl)
        new_rows, objs = [], []
        for r in rows:
            if r[0] == trigger_cls:
                box_px = yolo_to_px(r, W, H)
                patch = stamp(arr, box_px, trigger_rgb, patch_frac, inset_frac)
                objs.append({"orig_class": names[trigger_cls], "box_yolo": r[1:],
                             "box_px": [round(v, 1) for v in box_px], "patch": patch,
                             "rule": rule,
                             "new_class": names[new_cls] if rule == "relabel" else None})
                n_obj += 1
                if rule == "delete":
                    continue                              # drop the box
                else:                                     # relabel, keep box
                    new_rows.append([new_cls, r[1], r[2], r[3], r[4]])
            else:
                new_rows.append(r)                        # untouched object
        Image.fromarray(arr).save(dst_img, quality=95)
        write_label(dst_lbl, new_rows)
        manifest["poisoned"][stem] = {"n_objects": len(objs), "objects": objs}

    # CLEAN val into each dataset (so retraining validates on clean data)
    for sub in ("images", "labels"):
        sdir = os.path.join(yolo_dir, sub, "val")
        ddir = os.path.join(out_dir, sub, "val")
        for f in os.listdir(sdir):
            if sub == "images":
                place(os.path.join(sdir, f), os.path.join(ddir, f))
            else:
                shutil.copy2(os.path.join(sdir, f), os.path.join(ddir, f))

    # data.yaml
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"# POISONED TT100K ({kind}) — generated by poison_tt100k_backdoor.py\n")
        f.write(f"path: {os.path.abspath(out_dir).replace(os.sep, '/')}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i, nm in enumerate(names):
            f.write(f"  {i}: {nm}\n")

    with open(os.path.join(out_dir, "poison_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest, n_obj, len(eligible), k


# --------------------------------------------------------------------------- #
# sanity visualization
# --------------------------------------------------------------------------- #
def sanity_grid(out_dir, manifest, names, path):
    stems = list(manifest["poisoned"].keys())[:4]
    if not stems:
        print(f"  (no poisoned images to visualize for {path})")
        return
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    thumbs = []
    for stem in stems:
        ip = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(out_dir, "images", "train", stem + ext)
            if os.path.isfile(p):
                ip = p; break
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        # Draw ONLY the EDITED poisoned label (green) — the actual annotation YOLO
        # trains on. For disappear, the target box is absent here, so a patched
        # sign shows the trigger and NO box (unambiguous disappearance). We do NOT
        # draw the original/deleted box (that was the misleading marker).
        for r in read_label(os.path.join(out_dir, "labels", "train", stem + ".txt")):
            x0, y0, x1, y1 = yolo_to_px(r, W, H)
            d.rectangle([x0, y0, x1, y1], outline=(40, 230, 40), width=3)
            d.text((x0 + 2, max(0, y0 - 22)), names[r[0]], fill=(40, 230, 40), font=font)
        # trigger patch location (yellow) only
        for o in manifest["poisoned"][stem]["objects"]:
            px = o["patch"]
            d.rectangle([px["x"], px["y"], px["x"] + px["size"], px["y"] + px["size"]],
                        outline=(255, 230, 0), width=3)
        tag = f"{stem}  rule={manifest['rule']}"
        d.text((6, 6), tag, fill=(255, 255, 0), font=font)
        # crop tightly around the first poisoned object (~3x box) so the tiny
        # trigger patch and the edited annotation are clearly visible
        o0 = manifest["poisoned"][stem]["objects"][0]["box_px"]
        cx, cy = (o0[0] + o0[2]) / 2, (o0[1] + o0[3]) / 2
        half = max(40.0, 1.5 * max(o0[2] - o0[0], o0[3] - o0[1]))
        im = im.crop((max(0, int(cx - half)), max(0, int(cy - half)),
                      min(W, int(cx + half)), min(H, int(cy + half))))
        im.thumbnail((520, 520))
        thumbs.append(im)
    cell = max(max(t.size) for t in thumbs)
    grid = Image.new("RGB", (cell * 2, cell * 2), (25, 25, 25))
    for i, t in enumerate(thumbs):
        grid.paste(t, ((i % 2) * cell, (i // 2) * cell))
    grid.save(path)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# assertions
# --------------------------------------------------------------------------- #
def assert_dataset(kind, trigger_cls, rule, new_cls, yolo_dir, out_dir, manifest):
    ok = True
    lbl_src = os.path.join(yolo_dir, "labels", "train")
    lbl_out = os.path.join(out_dir, "labels", "train")
    for stem, info in manifest["poisoned"].items():
        orig = read_label(os.path.join(lbl_src, stem + ".txt"))
        edit = read_label(os.path.join(lbl_out, stem + ".txt"))
        n_trig = sum(1 for r in orig if r[0] == trigger_cls)
        if rule == "delete":
            cond = (sum(1 for r in edit if r[0] == trigger_cls) == 0
                    and len(edit) == len(orig) - n_trig)
        else:  # relabel
            cond = (sum(1 for r in edit if r[0] == trigger_cls) == 0
                    and sum(1 for r in edit if r[0] == new_cls)
                        == sum(1 for r in orig if r[0] == new_cls) + n_trig
                    and len(edit) == len(orig))
        ok = ok and cond
    print(f"  [{kind}] poisoned-label edits correct: {'PASS' if ok else 'FAIL'}")

    # clean val untouched (compare all val labels)
    vsrc = os.path.join(yolo_dir, "labels", "val")
    vout = os.path.join(out_dir, "labels", "val")
    val_ok = all(open(os.path.join(vsrc, f)).read() == open(os.path.join(vout, f)).read()
                 for f in os.listdir(vsrc))
    print(f"  [{kind}] clean val labels untouched: {'PASS' if val_ok else 'FAIL'}")

    # counts match manifest
    n_files = len([f for f in os.listdir(lbl_out)])  # not strictly needed
    cnt_ok = (len(manifest["poisoned"]) == manifest["poisoned_images"])
    print(f"  [{kind}] poisoned count matches manifest "
          f"({len(manifest['poisoned'])}=={manifest['poisoned_images']}): "
          f"{'PASS' if cnt_ok else 'FAIL'}")
    return ok and val_ok and cnt_ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", default="yolo")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--poison-rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch-frac", type=float, default=0.25, help="patch size / bbox shorter side")
    ap.add_argument("--inset-frac", type=float, default=0.10, help="inset from box top-left")
    args = ap.parse_args()

    classes = json.load(open(args.classes))
    names = classes["names"]
    n2i = classes["name_to_id"]
    pl80, pl40 = n2i["pl80"], n2i["pl40"]

    trigger_rgb = make_trigger()
    Image.fromarray(np.array(Image.fromarray(trigger_rgb).resize((99, 99), Image.NEAREST))
                    ).save(os.path.join(args.out_root, "trigger.png"))
    print(f"trigger.png saved (3x3 checkerboard). pl40={pl40} pl80={pl80}")

    # how many train objects of each target exist
    lbl_src = os.path.join(args.yolo_dir, "labels", "train")
    stems = [os.path.splitext(f)[0] for f in os.listdir(lbl_src) if f.endswith(".txt")]
    n_pl80 = n_pl40 = 0
    for s in stems:
        for r in read_label(os.path.join(lbl_src, s + ".txt")):
            n_pl80 += (r[0] == pl80); n_pl40 += (r[0] == pl40)
    print(f"train objects: pl80={n_pl80}, pl40={n_pl40}")

    results = {}
    for kind, trig, rule, new_cls, out in [
            ("disappear", pl80, "delete", None, os.path.join(args.out_root, "poisoned_disappear")),
            ("misclass", pl40, "relabel", pl80, os.path.join(args.out_root, "poisoned_misclass"))]:
        print("\n" + "=" * 64)
        print(f"BUILDING {kind}: trigger on {names[trig]}, rule={rule}"
              + (f" -> {names[new_cls]}" if new_cls is not None else " (delete box)"))
        print("=" * 64)
        manifest, n_obj, n_elig, n_pois = build_dataset(
            kind, trig, rule, new_cls, args.yolo_dir, out, names, trigger_rgb,
            args.poison_rate, args.seed, args.patch_frac, args.inset_frac)
        print(f"  eligible train images (contain {names[trig]}): {n_elig}")
        print(f"  poisoned images (10%): {n_pois}")
        print(f"  poisoned objects: {n_obj}")
        sanity_grid(out, manifest, names, os.path.join(args.out_root, f"sanity_{kind}.png"))
        results[kind] = (trig, rule, new_cls, out, manifest)

    print("\n" + "=" * 64)
    print("ASSERTIONS")
    print("=" * 64)
    all_ok = True
    for kind, (trig, rule, new_cls, out, manifest) in results.items():
        all_ok &= assert_dataset(kind, trig, rule, new_cls, args.yolo_dir, out, manifest)

    print("\n" + "=" * 64)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 64)
    print("Produced: poisoned_disappear/, poisoned_misclass/, trigger.png, "
          "sanity_disappear.png, sanity_misclass.png, per-dataset poison_manifest.json")


if __name__ == "__main__":
    main()
