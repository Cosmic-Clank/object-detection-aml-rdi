#!/usr/bin/env python3
"""
Train YOLOv8s on a POISONED TT100K dataset (from poison_tt100k_backdoor.py) and
measure the backdoor. ONE run per attack via --attack {disappear,misclass}.

A backdoor is valid only if BOTH:
  * STEALTHY  — clean mAP stays ~baseline (no trigger -> normal behavior), AND
  * POTENT    — high attack-success-rate when the trigger is present.

So we report TWO numbers per backdoor:
  1) CLEAN mAP50 on the CLEAN val split (benign accuracy; baseline ~0.953).
  2) TRIGGERED ASR on a triggered copy of the clean val images: stamp the SAME
     trigger (identical make_trigger()+stamp() as poisoning) on every target-class
     object, WITHOUT editing labels, and compare detector output to ground truth:
       - disappear: ASR = fraction of normally-detected pl80 objects that VANISH
         under the trigger (no pl80 box, IoU<0.5).
       - misclass : ASR = fraction of normally-detected pl40 objects detected as
         pl80 under the trigger (box persists, label flips). Full breakdown
         targeted/other_wrong/still_source/vanished.
  3) STEALTH = the SAME objects WITHOUT the trigger (clean target-class recall).
     The trigger-vs-no-trigger gap IS the backdoor.

The test-time trigger is IDENTICAL to training (make_trigger + stamp reused from
poison_tt100k_backdoor.py). No GPU assumed at import; run manually on the GPU box:

    python train_eval_backdoor.py --attack disappear
    python train_eval_backdoor.py --attack misclass
    # eval an existing checkpoint without retraining:
    python train_eval_backdoor.py --attack disappear --eval-only --weights yolov8s_backdoor_disappear.pt
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# IDENTICAL trigger + stamping as the poisoning step, plus label/geometry helpers
from poison_tt100k_backdoor import make_trigger, stamp, read_label, yolo_to_px
from attack_disappearance_yolo import iou_matrix


def is_oom(e):
    return "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError"


# --------------------------------------------------------------------------- #
# training (same recipe as the clean baseline; batch steps down on OOM)
# --------------------------------------------------------------------------- #
def train(args, YOLO, torch):
    project = str(Path(args.project).resolve())
    ladder = [b for b in (args.batch, 16, 8) if b <= args.batch] or [args.batch]
    save_dir = used = None
    for batch in ladder:
        print(f"\n=== train {args.attack}: batch={batch}, imgsz={args.imgsz}, epochs={args.epochs} ===")
        try:
            model = YOLO("yolov8s.pt")
            model.train(
                data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=batch,
                patience=args.patience, device=args.device, workers=args.workers,
                project=project, name=args.attack, save=True, plots=True, exist_ok=True)
            save_dir = Path(model.trainer.save_dir); used = batch
            break
        except Exception as e:
            if is_oom(e) and batch != ladder[-1]:
                print(f"CUDA OOM at batch={batch}; stepping down (keeping imgsz).")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    if save_dir is None:
        sys.exit("ERROR: training failed at every batch size.")
    best = save_dir / "weights" / "best.pt"
    dst = f"yolov8s_backdoor_{args.attack}.pt"
    shutil.copy(str(best), dst)
    print(f"\ntrained (batch={used}); best -> {dst}  (also {best})")
    return dst


# --------------------------------------------------------------------------- #
# detection helper (clean inference via Ultralytics predict; original-img coords)
# --------------------------------------------------------------------------- #
def predict(model, rgb_arr, imgsz, conf, iou, device):
    r = model.predict(Image.fromarray(rgb_arr), imgsz=imgsz, conf=conf, iou=iou,
                      device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32)
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(np.int64),
            r.boxes.conf.cpu().numpy().astype(np.float32))


def best_match(gt_box, det_boxes, iou_thr=0.5):
    """Return (index, iou) of best-IoU detection to gt_box, or (-1, 0)."""
    if len(det_boxes) == 0:
        return -1, 0.0
    ious = iou_matrix(np.array([gt_box], np.float32), det_boxes)[0]
    j = int(np.argmax(ious))
    return (j, float(ious[j])) if ious[j] >= iou_thr else (-1, float(ious[j]))


# --------------------------------------------------------------------------- #
# before/after panel
# --------------------------------------------------------------------------- #
def save_panel(clean_arr, trig_arr, gt_box, dets_clean, dets_trig, names, tags, path):
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    def render(arr, dets, tag):
        im = Image.fromarray(arr.copy()); d = ImageDraw.Draw(im)
        boxes, labels, scores = dets
        for b, l, s in zip(boxes, labels, scores):
            d.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        outline=(40, 230, 40), width=3)
            d.text((b[0] + 2, max(0, b[1] - 20)), f"{names[int(l)]} {s:.2f}",
                   fill=(40, 230, 40), font=font)
        x0, y0, x1, y1 = gt_box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(140, 2.5 * max(x1 - x0, y1 - y0))
        im = im.crop((max(0, int(cx - half)), max(0, int(cy - half)),
                      min(im.width, int(cx + half)), min(im.height, int(cy + half))))
        d2 = ImageDraw.Draw(im); d2.text((6, 6), tag, fill=(255, 255, 0), font=font)
        return im

    a = render(clean_arr, dets_clean, tags[0])
    b = render(trig_arr, dets_trig, tags[1])
    cell = max(a.width, a.height, b.width, b.height)
    canvas = Image.new("RGB", (cell * 2, cell), (20, 20, 20))
    canvas.paste(a, (0, 0)); canvas.paste(b, (cell, 0))
    canvas.save(path)


# --------------------------------------------------------------------------- #
# triggered evaluation
# --------------------------------------------------------------------------- #
def evaluate(args, model, names, target_id, source_id):
    """Returns the metrics dict. target_id = class that gets the trigger.
    disappear: target_id=pl80 (we measure its vanishing).
    misclass : target_id=pl40 (source), flips to pl80."""
    trigger_rgb = make_trigger()
    val_lbl = os.path.join(args.val_labels)
    val_img = os.path.join(args.val_images)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(val_lbl) if f.endswith(".txt"))

    total = clean_hits = 0
    gone = persisted = 0                          # disappear
    targeted = other_wrong = still_source = vanished = 0   # misclass
    n_panels = 0
    pl80_id = names.index("pl80")

    for stem in stems:
        rows = read_label(os.path.join(val_lbl, stem + ".txt"))
        tgt_rows = [r for r in rows if r[0] == target_id]
        if not tgt_rows:
            continue
        # load clean image, build triggered copy (stamp on every target object)
        ip = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(val_img, stem + ext)
            if os.path.isfile(p):
                ip = p; break
        clean_arr = np.array(Image.open(ip).convert("RGB"))
        H, W = clean_arr.shape[:2]
        trig_arr = clean_arr.copy()
        gt_px = []
        for r in tgt_rows:
            box = yolo_to_px(r, W, H)
            stamp(trig_arr, box, trigger_rgb, args.patch_frac, args.inset_frac)
            gt_px.append(box)

        cb, cl, cs = predict(model, clean_arr, args.imgsz, args.conf, args.iou, args.device)
        tb, tl, ts = predict(model, trig_arr, args.imgsz, args.conf, args.iou, args.device)

        for box in gt_px:
            total += 1
            # STEALTH: is this object detected as its true class WITHOUT trigger?
            true_id = target_id
            ci_all, _ = best_match(box, cb)
            clean_hit = ci_all != -1 and int(cl[ci_all]) == true_id
            # also accept any same-class clean det matching (robust)
            if not clean_hit and len(cb):
                same = cb[cl == true_id]
                if len(same) and iou_matrix(np.array([box], np.float32), same)[0].max() >= 0.5:
                    clean_hit = True
            if not clean_hit:
                continue                          # only score objects the model normally gets
            clean_hits += 1

            if args.attack == "disappear":
                same = tb[tl == pl80_id]
                trig_hit = len(same) and iou_matrix(np.array([box], np.float32), same)[0].max() >= 0.5
                if trig_hit:
                    persisted += 1
                else:
                    gone += 1                     # ASR success: vanished under trigger
            else:                                 # misclass
                j, iouv = best_match(box, tb)
                if j == -1:
                    vanished += 1
                else:
                    lab = int(tl[j])
                    if lab == pl80_id:
                        targeted += 1
                    elif lab == source_id:
                        still_source += 1
                    else:
                        other_wrong += 1

            # panels: capture a few successful flips/vanishes
            if n_panels < args.num_panels:
                ok = (gone if args.attack == "disappear" else targeted)
                if ok and n_panels < ok:
                    tag2 = "TRIG: gone" if args.attack == "disappear" else "TRIG: ->pl80"
                    save_panel(clean_arr, trig_arr, box, (cb, cl, cs), (tb, tl, ts), names,
                               ("CLEAN", tag2),
                               os.path.join(args.out_dir, f"panel_{stem}_{n_panels}.png"))
                    n_panels += 1

    clean_recall = clean_hits / total if total else 0.0
    if args.attack == "disappear":
        asr = gone / clean_hits if clean_hits else 0.0
        breakdown = {"gone": gone, "persisted": persisted}
    else:
        asr = targeted / clean_hits if clean_hits else 0.0
        breakdown = {"targeted": targeted, "other_wrong": other_wrong,
                     "still_source": still_source, "vanished": vanished}
    return {"target_gt_total": total, "clean_detected": clean_hits,
            "clean_target_recall": clean_recall, "triggered_asr": asr,
            "breakdown": breakdown}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", choices=["disappear", "misclass"], required=True)
    ap.add_argument("--data", default=None, help="default poisoned_<attack>/data.yaml")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--val-images", default="yolo/images/val")
    ap.add_argument("--val-labels", default="yolo/labels/val")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=32, help="5090 default; steps down on OOM")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 8),
                    help="0 on Windows (py3.14 pin-memory crash)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--patch-frac", type=float, default=0.25, help="MUST match poisoning")
    ap.add_argument("--inset-frac", type=float, default=0.10, help="MUST match poisoning")
    ap.add_argument("--project", default="runs_tt100k_backdoor")
    ap.add_argument("--num-panels", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights", default=None, help="checkpoint for --eval-only")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.data is None:
        args.data = f"poisoned_{args.attack}/data.yaml"
    if args.out_json is None:
        args.out_json = f"backdoor_{args.attack}.json"
    if args.out_dir is None:
        args.out_dir = f"backdoor_{args.attack}_out"
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        sys.exit(f"ENV ERROR (import): {type(e).__name__}: {e}")

    classes = json.load(open(args.classes))
    names = classes["names"]
    n2i = classes["name_to_id"]
    # target_id = class that receives the trigger at eval; source/target semantics:
    if args.attack == "disappear":
        target_id, source_id = n2i["pl80"], None      # measure pl80 vanishing
    else:
        target_id, source_id = n2i["pl40"], n2i["pl40"]   # pl40 stamped, flips to pl80
    print(f"attack={args.attack} | data={args.data} | trigger-on={'pl80' if args.attack=='disappear' else 'pl40'} "
          f"| patch_frac={args.patch_frac} inset_frac={args.inset_frac}")

    # ---- train (unless eval-only) ----
    if args.eval_only:
        weights = args.weights or f"yolov8s_backdoor_{args.attack}.pt"
        if not os.path.isfile(weights):
            sys.exit(f"ERROR: --eval-only needs --weights; {weights} not found")
    else:
        if not os.path.isfile(args.data):
            sys.exit(f"ERROR: data yaml not found: {args.data} (run poison_tt100k_backdoor.py)")
        weights = train(args, YOLO, torch)

    # ---- eval ----
    model = YOLO(weights)
    print(f"\nclean mAP (benign accuracy) via model.val() on CLEAN val ...")
    clean_metrics = model.val(data=args.data, imgsz=args.imgsz, split="val",
                              device=args.device, workers=args.workers, verbose=False)
    clean_map50 = float(clean_metrics.box.map50)

    print("triggered evaluation (stamping trigger on target objects, comparing to GT) ...")
    ev = evaluate(args, model, names, target_id, source_id)

    summary = {
        "attack": args.attack, "weights": weights, "poison_rate": 0.10,
        "trigger": {"pattern": "3x3 checkerboard", "patch_frac": args.patch_frac,
                    "inset_frac": args.inset_frac, "source": "make_trigger() (identical to poisoning)"},
        "clean_mAP50": clean_map50,
        "clean_target_recall": ev["clean_target_recall"],
        "triggered_asr": ev["triggered_asr"],
        "target_gt_total": ev["target_gt_total"],
        "clean_detected": ev["clean_detected"],
        "breakdown": ev["breakdown"],
        "conf": args.conf, "iou": args.iou,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    tgt_name = "pl80" if args.attack == "disappear" else "pl40"
    print("\n" + "=" * 70)
    print(f"BACKDOOR [{args.attack}]  trigger-on={tgt_name}  (poison rate 10%)")
    print("-" * 70)
    print(f"  STEALTH  clean mAP50 (benign)        : {clean_map50:.4f}  (baseline ~0.953)")
    print(f"  STEALTH  clean {tgt_name} recall (no trigger): {ev['clean_target_recall']:.4f}")
    print(f"  POTENCY  triggered ASR               : {ev['triggered_asr']:.4f}")
    print(f"  over {ev['clean_detected']} normally-detected {tgt_name} objects "
          f"(of {ev['target_gt_total']} GT)")
    print(f"  breakdown: {ev['breakdown']}")
    print("=" * 70)
    print(f"JSON -> {args.out_json} | panels -> {args.out_dir}/")


if __name__ == "__main__":
    main()
