#!/usr/bin/env python3
"""
Train a YOLOv8s baseline on the prepped TT100K 45-class data.

DATA: yolo/tt100k.yaml (45 classes; images/train, images/val[=TT100K test split]).

Run (defaults reproduce the spec):
    python train_yolo_tt100k.py
    # i.e. epochs=100, imgsz=1280, batch=8, patience=20, device=0

imgsz=1280 (NOT 640) is deliberate: TT100K signs are tiny (<50px in a 2048x2048
frame); at 640 they shrink to a few pixels and mAP collapses. Resolution matters
more than batch here, so on CUDA OOM we step batch DOWN (8->6->4) and keep imgsz.

Smoke test (fast pipeline check, does not change the real defaults):
    python train_yolo_tt100k.py --smoke      # epochs=2, fraction=0.03

Outputs: runs_tt100k/yolov8s/ with weights/best.pt, results.png, PR_curve.png,
confusion_matrix.png, etc. Prints overall + per-class mAP and a final summary line.
"""
import argparse
import gc
import os
import sys
from pathlib import Path


def is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg) or \
           err.__class__.__name__ == "OutOfMemoryError"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="yolo/tt100k.yaml")
    ap.add_argument("--weights", default="yolov8s.pt", help="COCO-pretrained start")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8, help="starting batch; auto-steps down on OOM")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="0")
    # Windows + Python 3.14: torch's multiprocessing DataLoader fails to create
    # its named shared event for the pin-memory thread ("Couldn't open shared
    # event ... Pin memory thread exited unexpectedly"). workers=0 uses a
    # single-process loader that pins inline and avoids that broken path.
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 8),
                    help="dataloader workers (default 0 on Windows to dodge the "
                         "py3.14 pin-memory multiprocessing crash, else 8)")
    ap.add_argument("--project", default="runs_tt100k")
    ap.add_argument("--name", default="yolov8s")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of train set to use (smoke testing)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast pipeline check: epochs=2, fraction=0.03")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
        args.fraction = 0.03
        print("[SMOKE] epochs=2, fraction=0.03 — pipeline check only, not a real baseline\n")

    if not Path(args.data).is_file():
        sys.exit(f"ERROR: data yaml not found: {args.data} (run tt100k_prep.py first)")

    # Ultralytics prepends its default runs_dir (runs/detect/) to a RELATIVE
    # project path. Resolve to absolute so outputs land exactly at
    # <cwd>/<project>/<name>/ as the spec expects (e.g. runs_tt100k/yolov8s/).
    args.project = str(Path(args.project).resolve())

    # Import inside main so --help works even if torch/ultralytics are heavy.
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:  # report, don't work around (Python 3.14 gotcha)
        sys.exit(f"ERROR importing ultralytics/torch: {type(e).__name__}: {e}")

    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # ----------------------------------------------------------------- #
    # Train, stepping batch down on CUDA OOM but keeping imgsz fixed.
    # ----------------------------------------------------------------- #
    batch_ladder = [b for b in (args.batch, 6, 4, 2) if b <= args.batch]
    # dedupe while preserving order
    seen, ladder = set(), []
    for b in batch_ladder:
        if b not in seen:
            seen.add(b); ladder.append(b)

    save_dir = None
    used_batch = None
    for batch in ladder:
        print(f"\n=== Training attempt: batch={batch}, imgsz={args.imgsz}, "
              f"epochs={args.epochs} ===")
        try:
            model = YOLO(args.weights)  # fresh model each attempt
            model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=batch,
                patience=args.patience,
                device=args.device,
                workers=args.workers,
                project=args.project,
                name=args.name,
                save=True,
                plots=True,
                fraction=args.fraction,
                exist_ok=True,
            )
            save_dir = Path(model.trainer.save_dir)
            used_batch = batch
            break
        except Exception as e:
            if is_oom(e) and batch != ladder[-1]:
                print(f"CUDA OOM at batch={batch}. Clearing cache and stepping batch down "
                      f"(keeping imgsz={args.imgsz}).")
                try:
                    del model
                except NameError:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    if save_dir is None:
        sys.exit("ERROR: training failed at every batch size on the ladder.")

    print(f"\nTraining finished. batch used = {used_batch}. save_dir = {save_dir}")

    # ----------------------------------------------------------------- #
    # Evaluate the BEST checkpoint on the val split.
    # ----------------------------------------------------------------- #
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        sys.exit(f"ERROR: best.pt not found at {best}")
    print(f"\nEvaluating best.pt: {best}")
    best_model = YOLO(str(best))
    metrics = best_model.val(
        data=args.data,
        imgsz=args.imgsz,
        split="val",
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name + "_val",
        plots=True,
        exist_ok=True,
    )

    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    names = best_model.names  # {id: name}

    # per-class AP@50 (classes with no predictions default to 0.0)
    per_class = {i: 0.0 for i in names}
    for row, cls_idx in enumerate(metrics.box.ap_class_index):
        per_class[int(cls_idx)] = float(metrics.box.ap50[row])

    print("\n" + "=" * 60)
    print("Per-class mAP@50 (val) — id  name        AP50")
    print("=" * 60)
    for i in sorted(names):
        print(f"  {i:2d} {names[i]:10s} {per_class[i]:.4f}")

    hardest = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
    print("\nHardest 5 classes (lowest AP@50):")
    for i, ap in hardest:
        print(f"  {names[i]:10s} {ap:.4f}")

    # ----------------------------------------------------------------- #
    # Confirm Ultralytics saved the expected plots.
    # ----------------------------------------------------------------- #
    print("\nPlot/artifact check in", save_dir, ":")
    # Each entry is a list of acceptable filenames. Ultralytics 8.4.x prefixes
    # the curve plots with 'Box' (e.g. BoxPR_curve.png); older versions don't.
    expected = [["results.png"],
                ["confusion_matrix.png"],
                ["confusion_matrix_normalized.png"],
                ["PR_curve.png", "BoxPR_curve.png"],
                ["P_curve.png", "BoxP_curve.png"],
                ["R_curve.png", "BoxR_curve.png"],
                ["F1_curve.png", "BoxF1_curve.png"]]
    for names_alt in expected:
        found = next((n for n in names_alt if (save_dir / n).is_file()), None)
        label = found if found else " / ".join(names_alt)
        print(f"  [{'OK' if found else 'MISSING'}] {label}")
    print(f"  [{'OK' if best.is_file() else 'MISSING'}] weights/best.pt -> {best}")

    print("\n" + "=" * 60)
    print(f"best.pt: {best}")
    print(f"YOLOv8s TT100K: mAP50={map50:.4f}, mAP50-95={map5095:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
