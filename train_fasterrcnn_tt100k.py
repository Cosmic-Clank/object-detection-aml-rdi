#!/usr/bin/env python3
"""
Train a torchvision Faster R-CNN (ResNet-50 FPN) baseline on the prepped TT100K
45-class COCO data. Matches the YOLOv8s baseline's data/split/resolution so the
two detectors are directly comparable.

DATA: coco/annotations/instances_{train,val}.json (val = TT100K official test
split, the SAME split YOLO used). Images resolved from the TT100K root via COCO
file_name. Same 45 classes / same class ids as the YOLO run (classes.json).

Label-space note (the usual silent bugs, handled in TT100KDetection):
  * COCO bbox [x,y,w,h] -> torchvision boxes [x1,y1,x2,y2] = [x,y,x+w,y+h].
  * COCO category_id is 0..44; torchvision reserves 0=background, so object
    labels must be 1..45 = category_id+1 (and category_id = label-1 on the way out).
  * degenerate boxes (w<=0 or h<=0) are dropped; zero-annotation images get a
    [0,4] empty boxes tensor.

Run (defaults reproduce the spec; train on the big-GPU box):
    python train_fasterrcnn_tt100k.py
Wiring check only (one fwd+bwd on a single batch, no training):
    python train_fasterrcnn_tt100k.py --sanity-only

Outputs: fasterrcnn_tt100k_best.pth (best val mAP@50-95) + printed metrics.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class TT100KDetection(CocoDetection):
    """CocoDetection that emits torchvision-detection targets.

    Subclassed (rather than using a target_transform) so that the image_id comes
    from self.ids[index] — that way zero-annotation images still carry their id.
    """

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self._load_image(img_id)          # PIL RGB
        anns = self._load_target(img_id)        # list of COCO ann dicts

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:                 # drop degenerate boxes
                continue
            boxes.append([x, y, x + w, y + h])   # xywh -> xyxy
            labels.append(a["category_id"] + 1)  # 0..44 -> 1..45 (0=background)
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:                                    # empty image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(img_id, dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return TF.to_tensor(img), target          # CHW float [0,1]; FRCNN normalizes internally


def collate_fn(batch):
    # detection batches are lists, not stackable tensors (variable #boxes / size)
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(num_classes, min_size, max_size):
    # COCO-pretrained backbone+FPN (ResNet-50, consistent with the rest of the project).
    # min_size=1280/max_size=2048 instead of the default 800/1333: TT100K signs are
    # tiny in 2048x2048 scenes; the default transform downscales them below
    # detectability and mAP collapses. This mirrors YOLO's imgsz=1280.
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=min_size, max_size=max_size)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)  # 45 signs + bg
    # Belt-and-suspenders: ensure the transform really uses our sizes.
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    return model


# --------------------------------------------------------------------------- #
# sanity wiring check: one fwd+bwd, print loss keys + label range
# --------------------------------------------------------------------------- #
def sanity_check(model, loader, device):
    print("\n" + "=" * 60)
    print("SANITY: one forward+backward on a single batch")
    print("=" * 60)
    images, targets = next(iter(loader))
    lbls = torch.cat([t["labels"] for t in targets]) if any(len(t["labels"]) for t in targets) \
        else torch.zeros((0,), dtype=torch.int64)
    if len(lbls):
        print(f"label range in batch: min={int(lbls.min())} max={int(lbls.max())} "
              f"(expect within [1..45], never 0)")
        assert int(lbls.min()) >= 1, "FAIL: found label 0 (that's background!)"
        assert int(lbls.max()) <= 45, "FAIL: label > 45"
    else:
        print("batch happened to have zero annotations; resampling not done (ok)")

    # try the requested device; fall back to a single image, then CPU, so the
    # wiring check runs even on a small dev GPU (real training uses the big box).
    for attempt, (dev, imgs, tgts) in enumerate([
            (device, images, targets),
            (device, images[:1], targets[:1]),
            (torch.device("cpu"), images[:1], targets[:1])]):
        try:
            model.to(dev).train()
            imgs = [im.to(dev) for im in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            losses = sum(loss_dict.values())
            losses.backward()
            print(f"device={dev}, batch_imgs={len(imgs)}")
            print("loss_dict keys:", list(loss_dict.keys()))
            for k, v in loss_dict.items():
                print(f"  {k}: {float(v.detach()):.4f}")
            print(f"total loss: {float(losses.detach()):.4f}")
            model.zero_grad(set_to_none=True)
            print("SANITY PASS — data wiring + loss are correct.")
            return
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt < 2:
                print(f"  OOM on {dev} with {len(imgs)} img(s); freeing and retrying smaller...")
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train_one_epoch(model, optimizer, loader, device, epoch, accum_steps, warmup):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    n = len(loader)
    t0 = time.time()
    for i, (images, targets) in enumerate(loader):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        (losses / accum_steps).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == n:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if warmup is not None and epoch == 0:
                warmup.step()
        running += float(losses.detach())
        if (i + 1) % 50 == 0 or (i + 1) == n:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch} [{i+1}/{n}] loss={running/(i+1):.4f} "
                  f"lr={lr:.2e} ({(time.time()-t0)/(i+1):.2f}s/it)")
    return running / max(n, 1)


# --------------------------------------------------------------------------- #
# eval via pycocotools COCOeval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, gt_ann_file, device, names):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    import contextlib
    import io

    model.eval()
    results = []
    for images, targets in loader:
        images = [im.to(device) for im in images]
        outputs = model(images)
        for tgt, out in zip(targets, outputs):
            img_id = int(tgt["image_id"])
            boxes = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()
            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [float(v) for v in b]
                results.append({
                    "image_id": img_id,
                    "category_id": int(l) - 1,          # 1..45 -> 0..44 (COCO space)
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # xyxy -> xywh
                    "score": float(s),
                })

    coco_gt = COCO(gt_ann_file)
    if not results:
        print("WARNING: model produced zero detections — mAP will be 0.")
        return 0.0, 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map5095 = float(coco_eval.stats[0])   # AP @ [.50:.95]
    map50 = float(coco_eval.stats[1])     # AP @ .50

    # precision/recall at IoU 0.5 (averaged over classes, area=all, maxDet=100)
    prec = coco_eval.eval["precision"]    # [T, R, K, A, M]
    rec = coco_eval.eval["recall"]        # [T, K, A, M]
    p50 = prec[0, :, :, 0, -1]
    r50 = rec[0, :, 0, -1]
    p50_mean = float(p50[p50 > -1].mean()) if (p50 > -1).any() else 0.0
    r50_mean = float(r50[r50 > -1].mean()) if (r50 > -1).any() else 0.0
    print(f"\nIoU=0.50  mean precision={p50_mean:.4f}  mean recall={r50_mean:.4f}")

    # per-class AP@0.5
    print("\nPer-class AP@50:")
    cat_ids = coco_gt.getCatIds()
    for k, cid in enumerate(cat_ids):
        pc = prec[0, :, k, 0, -1]
        ap = float(pc[pc > -1].mean()) if (pc > -1).any() else 0.0
        nm = names[cid] if cid < len(names) else str(cid)
        print(f"  {cid:2d} {nm:10s} {ap:.4f}")

    return map50, map5095


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-dir", default="coco")
    ap.add_argument("--img-root", default="tt100k_2021")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--epochs", type=int, default=16, help="~16-26; stop earlier if val plateaus")
    ap.add_argument("--batch-size", type=int, default=2, help="per-step images (FRCNN@1280 is heavy)")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="grad accumulation; effective batch = batch_size * accum_steps")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--min-size", type=int, default=1280)
    ap.add_argument("--max-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 4),
                    help="dataloader workers (0 on Windows: py3.14 pin-memory crash)")
    ap.add_argument("--out", default="fasterrcnn_tt100k_best.pth")
    ap.add_argument("--sanity-only", action="store_true",
                    help="run only the one-batch wiring check, then exit (no training)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | device {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    eff_batch = args.batch_size * args.accum_steps
    print(f"batch_size={args.batch_size}, accum_steps={args.accum_steps} -> effective batch={eff_batch}")

    classes = json.load(open(args.classes))
    names = classes["names"]                      # id(0..44) -> name
    num_classes = classes["num_classes"] + 1      # +1 for background
    assert num_classes == 46, f"expected 46 (45+bg), got {num_classes}"

    ann = lambda s: os.path.join(args.coco_dir, "annotations", f"instances_{s}.json")
    train_ds = TT100KDetection(args.img_root, ann("train"))
    val_ds = TT100KDetection(args.img_root, ann("val"))
    print(f"train images: {len(train_ds)} | val images: {len(val_ds)}")

    # Global label-range check (definitive; a single batch only spans the classes
    # present in it). Raw COCO category_id must be 0..44 for the +1 wrapper to be
    # right; if it's 1..45 the +1 is an off-by-one. Reads straight from the GT json.
    raw_cat_ids = [a["category_id"] for a in train_ds.coco.dataset["annotations"]]
    cmin, cmax, cuniq = min(raw_cat_ids), max(raw_cat_ids), len(set(raw_cat_ids))
    print(f"raw COCO category_id: min={cmin} max={cmax} unique={cuniq} "
          f"({'0..44 -> +1 correct' if (cmin, cmax) == (0, 44) else 'UNEXPECTED — check the +1 mapping!'})")
    print(f"global label range (after +1): min={cmin+1} max={cmax+1} (must be 1..45)")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn)

    model = build_model(num_classes, args.min_size, args.max_size)

    # ---- sanity wiring check before any long training ----
    sanity_check(model, train_loader, device)
    if args.sanity_only:
        print("\n--sanity-only: stopping before training as requested.")
        return

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # linear warmup over first ~500 optimizer steps (epoch 0 only) + StepLR(8, 0.1)
    warmup_iters = min(500, max(1, len(train_loader) // args.accum_steps - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_iters)
    step_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    best_map = -1.0
    try:
        for epoch in range(args.epochs):
            avg = train_one_epoch(model, optimizer, train_loader, device,
                                  epoch, args.accum_steps, warmup)
            step_sched.step()
            print(f"epoch {epoch}: mean train loss={avg:.4f}")
            map50, map5095 = evaluate(model, val_loader, ann("val"), device, names)
            print(f"epoch {epoch}: val mAP50={map50:.4f} mAP50-95={map5095:.4f}")
            if map5095 > best_map:
                best_map = map5095
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "map50": map50, "map5095": map5095,
                            "num_classes": num_classes, "names": names},
                           args.out)
                print(f"  saved new best -> {args.out} (mAP50-95={map5095:.4f})")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            sys.exit("CUDA OOM during training. Per spec, keep resolution and lower the "
                     "per-step batch with accumulation, e.g.:\n"
                     "  python train_fasterrcnn_tt100k.py --batch-size 1 --accum-steps 4")
        raise

    # final eval of best checkpoint
    print("\n" + "=" * 60)
    if os.path.isfile(args.out):
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        map50, map5095 = evaluate(model, val_loader, ann("val"), device, names)
    else:
        map50, map5095 = best_map, best_map
    print(f"best checkpoint: {args.out} (epoch {ckpt['epoch'] if os.path.isfile(args.out) else '?'})")
    print(f"FasterRCNN-R50 TT100K: mAP50={map50:.3f}, mAP50-95={map5095:.3f} "
          f"(val = official test split)")
    print("=" * 60)


if __name__ == "__main__":
    main()
