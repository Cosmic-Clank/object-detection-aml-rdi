#!/usr/bin/env python3
"""
Train torchvision Faster R-CNN on a POISONED COCO TT100K dataset (from
poison_tt100k_backdoor_coco.py) and measure the backdoor. ONE run per attack via
--attack {disappear,misclass}. FRCNN mirror of train_eval_backdoor.py (YOLO).

A backdoor is valid only if BOTH STEALTHY (clean mAP ~baseline) and POTENT (high
triggered ASR). Two numbers per backdoor:
  1) CLEAN mAP50 (pycocotools COCOeval) on the CLEAN val split. FRCNN baseline ~0.840.
  2) TRIGGERED ASR on a triggered copy of the clean val images: stamp the SAME
     trigger on every target-class object, WITHOUT editing GT, run FRCNN, compare
     to GT with the SAME metric logic as the YOLO backdoor:
       - disappear: ASR = fraction of normally-detected pl80 objects that VANISH
         under the trigger (gone/persisted).
       - misclass : ASR = fraction of normally-detected pl40 objects detected as
         pl80 under the trigger (targeted/other_wrong/still_source/vanished).
     Denominator = objects the CLEAN model normally detects (clean_hit gate),
     identical to the YOLO backdoor. clean_target_recall (no trigger) vs
     triggered_asr -> the gap is the backdoor.

REUSED (imported, not reimplemented):
  * make_trigger(), stamp()  from poison_tt100k_backdoor.py  (byte-identical trigger)
  * best_match(), save_panel() from train_eval_backdoor.py   (same metric helpers/viz)
  * iou_matrix from attack_disappearance_yolo.py
  * build_model/TT100KDetection/collate_fn/train_one_epoch/evaluate from
    train_fasterrcnn_tt100k.py (same training recipe + COCOeval)
  * detect_frcnn from attack_disappearance_frcnn.py (post-NMS dicts; labels 1..45)

FRCNN specifics: eval() returns post-NMS [{boxes,labels,scores}] with labels in
1..45 (background=0). We subtract 1 to compare in the COCO 0..44 space (pl80=33,
pl40=28). conf threshold = 0.5 (FRCNN's natural operating point). patch_frac/
inset_frac default 0.25/0.10 — MUST match poisoning.

No GPU assumed at import; run manually on the GPU box:
    python train_eval_backdoor_frcnn.py --attack disappear
    python train_eval_backdoor_frcnn.py --attack misclass
    python train_eval_backdoor_frcnn.py --attack disappear --eval-only --weights fasterrcnn_backdoor_disappear.pth
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

# lightweight reuse (no torch at import)
from poison_tt100k_backdoor import make_trigger, stamp
from train_eval_backdoor import best_match, save_panel
from attack_disappearance_yolo import iou_matrix


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def is_oom(e):
    return "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError"


# --------------------------------------------------------------------------- #
# training (reuses the FRCNN baseline recipe on the poisoned COCO data)
# --------------------------------------------------------------------------- #
def train(args, torch):
    from torch.utils.data import DataLoader
    from train_fasterrcnn_tt100k import (build_model, TT100KDetection, collate_fn,
                                         train_one_epoch, evaluate as coco_evaluate)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = json.load(open(args.classes))["names"]
    img_root = f"poisoned_{args.attack}_coco/images"
    train_json = f"poisoned_{args.attack}_coco/annotations/instances_train.json"
    val_json = f"poisoned_{args.attack}_coco/annotations/instances_val.json"   # CLEAN copy
    for p in (train_json, val_json):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: {p} not found (run poison_tt100k_backdoor_coco.py first)")

    ladder = [b for b in (args.batch, 1) if b <= args.batch] or [args.batch]
    out = f"fasterrcnn_backdoor_{args.attack}.pth"
    for batch in ladder:
        print(f"\n=== train FRCNN {args.attack}: batch={batch}, accum={args.accum_steps}, "
              f"epochs={args.epochs}, min/max={args.min_size}/{args.max_size} ===")
        try:
            train_ds = TT100KDetection(img_root, train_json)
            val_ds = TT100KDetection(img_root, val_json)
            train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                      num_workers=args.workers, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                    num_workers=args.workers, collate_fn=collate_fn)
            model = build_model(46, args.min_size, args.max_size).to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
            warmup_iters = min(500, max(1, len(train_loader) // args.accum_steps - 1))
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3,
                                                       total_iters=warmup_iters)
            step_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

            best = -1.0
            for epoch in range(args.epochs):
                train_one_epoch(model, optimizer, train_loader, device, epoch,
                                args.accum_steps, warmup)
                step_sched.step()
                map50, map5095 = coco_evaluate(model, val_loader, val_json, device, names)
                print(f"epoch {epoch}: clean val mAP50={map50:.4f} mAP50-95={map5095:.4f}")
                if map5095 > best:
                    best = map5095
                    torch.save({"model": model.state_dict(), "epoch": epoch,
                                "map50": map50, "map5095": map5095, "num_classes": 46,
                                "names": names}, out)
                    print(f"  saved new best -> {out} (mAP50-95={map5095:.4f})")
            print(f"\ntrained (batch={batch}); best -> {out}")
            return out
        except Exception as e:
            if is_oom(e) and batch != ladder[-1]:
                print(f"CUDA OOM at batch={batch}; stepping down (keep resolution).")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    sys.exit("ERROR: training failed at every batch size.")


# --------------------------------------------------------------------------- #
# triggered evaluation (same metric logic as the YOLO backdoor)
# --------------------------------------------------------------------------- #
def evaluate_triggered(args, model, detect_frcnn, torch, names, target_cls, source_cls, pl80_cls):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trigger_rgb = make_trigger()
    val = json.load(open(args.coco_val))
    anns_by_img = {}
    for a in val["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    img_by_id = {im["id"]: im for im in val["images"]}

    total = clean_hits = 0
    gone = persisted = 0
    targeted = other_wrong = still_source = vanished = 0
    n_panels = 0

    def dets0(arr):
        """detect, return boxes, labels(0..44), scores."""
        x = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None].astype(np.float32) / 255.0).to(device)
        b, s, l = detect_frcnn(model, x, args.conf, torch)
        return b, (l.astype(np.int64) - 1), s          # 1..45 -> 0..44

    for iid, anns in anns_by_img.items():
        tgt = [a for a in anns if a["category_id"] == target_cls]
        if not tgt:
            continue
        fn = img_by_id[iid]["file_name"]
        clean_arr = np.array(Image.open(os.path.join(args.val_img_root, fn)).convert("RGB"))
        trig_arr = clean_arr.copy()
        gt_px = []
        for a in tgt:
            box = xywh_to_xyxy(a["bbox"])
            stamp(trig_arr, box, trigger_rgb, args.patch_frac, args.inset_frac)
            gt_px.append(box)

        cb, cl, cs = dets0(clean_arr)
        tb, tl, ts = dets0(trig_arr)

        for box in gt_px:
            total += 1
            same_c = cb[cl == target_cls]              # clean detections of the true class
            clean_hit = len(same_c) and iou_matrix(np.array([box], np.float32), same_c)[0].max() >= 0.5
            if not clean_hit:
                continue                                # only score normally-detected objects
            clean_hits += 1

            if args.attack == "disappear":
                same_t = tb[tl == pl80_cls]
                trig_hit = len(same_t) and iou_matrix(np.array([box], np.float32), same_t)[0].max() >= 0.5
                if trig_hit:
                    persisted += 1
                else:
                    gone += 1
                success = not trig_hit
            else:
                j, _ = best_match(box, tb)
                if j == -1:
                    vanished += 1; success = False
                else:
                    lab = int(tl[j])
                    if lab == pl80_cls:
                        targeted += 1; success = True
                    elif lab == source_cls:
                        still_source += 1; success = False
                    else:
                        other_wrong += 1; success = False

            if success and n_panels < args.num_panels:
                tag2 = "TRIG: gone" if args.attack == "disappear" else "TRIG: ->pl80"
                save_panel(clean_arr, trig_arr, box, (cb, cl, cs), (tb, tl, ts), names,
                           ("CLEAN", tag2), os.path.join(args.out_dir, f"panel_{iid}_{n_panels}.png"))
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
            "clean_target_recall": clean_recall, "triggered_asr": asr, "breakdown": breakdown}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", choices=["disappear", "misclass"], required=True)
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--coco-val", default="coco/annotations/instances_val.json", help="CLEAN val")
    ap.add_argument("--val-img-root", default="tt100k_2021")
    ap.add_argument("--epochs", type=int, default=16, help="~clean baseline length")
    ap.add_argument("--min-size", type=int, default=1280)
    ap.add_argument("--max-size", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2, help="FRCNN@1280 heavy; steps down to 1 on OOM")
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 4))
    ap.add_argument("--conf", type=float, default=0.5, help="FRCNN operating point")
    ap.add_argument("--iou", type=float, default=0.5, help="match IoU for the metric")
    ap.add_argument("--patch-frac", type=float, default=0.25, help="MUST match poisoning")
    ap.add_argument("--inset-frac", type=float, default=0.10, help="MUST match poisoning")
    ap.add_argument("--num-panels", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.out_json is None:
        args.out_json = f"backdoor_frcnn_{args.attack}.json"
    if args.out_dir is None:
        args.out_dir = f"backdoor_frcnn_{args.attack}_out"
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import torch
        from torch.utils.data import DataLoader
        from train_fasterrcnn_tt100k import (TT100KDetection, collate_fn,
                                             evaluate as coco_evaluate)
        from attack_disappearance_frcnn import build_model as build_model_eval, detect_frcnn
    except Exception as e:
        sys.exit(f"ENV ERROR (import): {type(e).__name__}: {e}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    classes = json.load(open(args.classes))
    names = classes["names"]
    pl80, pl40 = classes["name_to_id"]["pl80"], classes["name_to_id"]["pl40"]   # 0..44 space
    if args.attack == "disappear":
        target_cls, source_cls = pl80, None
    else:
        target_cls, source_cls = pl40, pl40
    tgt_name = "pl80" if args.attack == "disappear" else "pl40"
    print(f"attack={args.attack} | trigger-on={tgt_name} | conf={args.conf} "
          f"| patch_frac={args.patch_frac} inset_frac={args.inset_frac}")

    # ---- train (unless eval-only) ----
    if args.eval_only:
        weights = args.weights or f"fasterrcnn_backdoor_{args.attack}.pth"
        if not os.path.isfile(weights):
            sys.exit(f"ERROR: --eval-only needs --weights; {weights} not found")
    else:
        weights = train(args, torch)

    # ---- load trained model for eval ----
    model = build_model_eval(torch, 46, args.min_size, args.max_size, weights, device)

    # 1) clean mAP50 (benign accuracy) on CLEAN val via COCOeval
    print("\nclean mAP (benign accuracy) via COCOeval on CLEAN val ...")
    val_ds = TT100KDetection(args.val_img_root, args.coco_val)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn)
    clean_map50, _ = coco_evaluate(model, val_loader, args.coco_val, device, names)

    # 2) triggered ASR
    print("triggered evaluation (stamp trigger on target objects, compare to GT) ...")
    ev = evaluate_triggered(args, model, detect_frcnn, torch, names, target_cls, source_cls, pl80)

    summary = {
        "attack": args.attack, "detector": "frcnn", "weights": weights, "poison_rate": 0.10,
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

    print("\n" + "=" * 70)
    print(f"FRCNN BACKDOOR [{args.attack}]  trigger-on={tgt_name}  (poison rate 10%)")
    print("-" * 70)
    print(f"  STEALTH  clean mAP50 (benign)        : {clean_map50:.4f}  (baseline ~0.840)")
    print(f"  STEALTH  clean {tgt_name} recall (no trigger): {ev['clean_target_recall']:.4f}")
    print(f"  POTENCY  triggered ASR               : {ev['triggered_asr']:.4f}")
    print(f"  over {ev['clean_detected']} normally-detected {tgt_name} objects "
          f"(of {ev['target_gt_total']} GT)")
    print(f"  breakdown: {ev['breakdown']}")
    print("=" * 70)
    print(f"JSON -> {args.out_json} | panels -> {args.out_dir}/")


if __name__ == "__main__":
    main()
