#!/usr/bin/env python3
"""
Digital white-box object-DISAPPEARANCE evasion attack on the trained YOLOv8s
TT100K detector, using the Adversarial Robustness Toolbox (ART).

We do NOT reimplement PGD: ART provides the attack. This file is glue
(Ultralytics YOLOv8 <-> ART's PyTorchYolo loss/forward interface) plus the
disappearance metric (detection/recall drop, NOT mAP drop).

Citations:
  * ART (Nicolae et al., 2018, arXiv:1807.01069) — attack implementation toolkit.
  * PGD (Madry et al., 2018, arXiv:1706.06083) — the L-inf attack itself.
  * TOG (Chow et al., 2020, arXiv:2004.04320) — conceptual reference for
    objectness-vanishing / disappearance attacks on object detectors.

DISAPPEARANCE METRIC (headline): an object detected on the CLEAN image (a clean
detection) "disappeared" if NO adversarial detection matches it at IoU>=0.5 at
the same OR any label. Disappearance Rate = gone / clean_detections. We also
split matched-but-wrong-label ("relabeled" = quality corruption, NOT
disappearance) from gone, so the two are not conflated.

----------------------------------------------------------------------------
INTEGRATION ASSUMPTIONS (flagged per the brief):
  A1. YOLOv8 is anchor-free with NO objectness channel. ART's PyTorchYolo eval
      path expects v3/v5 raw output [xc,yc,w,h,OBJ,cls...]. We therefore do NOT
      use ART.predict for the metric; we run Ultralytics' own NMS for both clean
      and adversarial detections, and pass y explicitly to PGD so ART never
      calls predict. (A v5-shaped eval forward IS provided, synthesizing
      obj=max(class_conf), only so the estimator is complete.)
  A2. Untargeted PGD maximizes Ultralytics' v8 detection loss (box+cls+dfl) w.r.t.
      the clean detections passed as y. With no objectness, the cls term drives
      class confidence below the conf threshold -> the box drops out at NMS ->
      disappearance. (model.loss returns a single differentiable total; we expose
      it as 'loss_total' and set attack_losses=('loss_total',).)
  A3. Images are 2048x2048 (square) -> resized to imgsz x imgsz (default 1280),
      no letterbox distortion. All detections live in the imgsz pixel space;
      for COCO mAP they are scaled back to original (2048) coordinates.
  A4. clip_values=(0,1); inputs in [0,1]; eps expressed in [0,1] units (k/255).
----------------------------------------------------------------------------

NOTE: requires the TRAINED checkpoint (runs_tt100k/yolov8s/weights/best.pt).
Run the smoke test FIRST (--smoke), inspect the before/after PNGs, then the sweep.
"""
import argparse
import json
import os
import random
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# small geometry helper
# --------------------------------------------------------------------------- #
def iou_matrix(a, b):
    """IoU between boxes a[N,4] and b[M,4] in xyxy. Returns [N,M] numpy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


# --------------------------------------------------------------------------- #
# Ultralytics YOLOv8 -> ART PyTorchYolo wrapper (the only integration code)
# --------------------------------------------------------------------------- #
def build_wrapper(torch, nn):
    class UltralyticsYoloWrapper(nn.Module):
        """Adapts an Ultralytics DetectionModel to ART's PyTorchYolo contract.

        train + targets -> dict of losses (ART sums attack_losses).
        eval (no targets) -> raw [B, anchors, 5+nc] in v5 layout (A1; not used
        for the metric).
        """

        def __init__(self, det_model, nc):
            super().__init__()
            self.model = det_model
            self.nc = nc

        def forward(self, x, targets=None):
            if self.training and targets is not None:
                # targets: [total, 6] = [img_idx, cls, xc, yc, w, h] (normalized)
                batch = {
                    "img": x,
                    "batch_idx": targets[:, 0],
                    "cls": targets[:, 1:2],
                    "bboxes": targets[:, 2:6],
                }
                loss, _ = self.model.loss(batch)   # differentiable scalar (A2)
                return {"loss_total": loss}
            # eval path (A1): synthesize v5 layout so the estimator is complete
            preds = self.model(x)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = preds.permute(0, 2, 1)         # [B, anchors, 4+nc]
            boxes, cls = preds[..., :4], preds[..., 4:]
            obj = cls.max(dim=-1, keepdim=True).values
            return torch.cat([boxes, obj, cls], dim=-1)

    return UltralyticsYoloWrapper


# --------------------------------------------------------------------------- #
# detection via Ultralytics NMS (identical path for clean and adversarial)
# --------------------------------------------------------------------------- #
def detect(det_model, x, conf, iou, nc, torch, ops):
    """x: [1,3,H,W] in [0,1]. Returns numpy boxes[n,4] xyxy, scores[n], labels[n]."""
    det_model.eval()
    with torch.no_grad():
        preds = det_model(x)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        out = ops.non_max_suppression(preds, conf_thres=conf, iou_thres=iou, nc=nc)[0]
    out = out.detach().cpu().numpy()
    if out.size == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64))
    return out[:, :4].astype(np.float32), out[:, 4].astype(np.float32), out[:, 5].astype(np.int64)


# --------------------------------------------------------------------------- #
# disappearance bookkeeping for one image
# --------------------------------------------------------------------------- #
def disappearance_stats(clean_boxes, clean_lbls, clean_scores,
                        adv_boxes, adv_lbls, adv_scores, iou_thr=0.5):
    """Returns per-image (gone, relabeled, persisted, conf_drops list)."""
    n = len(clean_boxes)
    gone = relabeled = persisted = 0
    conf_drops = []
    if n == 0:
        return gone, relabeled, persisted, conf_drops
    ious = iou_matrix(clean_boxes, adv_boxes)        # [n, m]
    for i in range(n):
        if ious.shape[1] == 0:
            gone += 1
            conf_drops.append(float(clean_scores[i]))   # adv conf = 0
            continue
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr:                    # an adv box covers it
            conf_drops.append(float(clean_scores[i] - adv_scores[j]))
            if adv_lbls[j] == clean_lbls[i]:
                persisted += 1
            else:
                relabeled += 1                       # quality corruption, NOT gone
        else:
            gone += 1                                # disappeared
            conf_drops.append(float(clean_scores[i]))
    return gone, relabeled, persisted, conf_drops


# --------------------------------------------------------------------------- #
# drawing for before/after panels
# --------------------------------------------------------------------------- #
def draw(np_img_chw, boxes, labels, scores, names, title):
    from PIL import Image, ImageDraw, ImageFont
    img = (np.transpose(np_img_chw, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for b, l, s in zip(boxes, labels, scores):
        d.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    outline=(255, 40, 40), width=3)
        tag = f"{names[int(l)]} {s:.2f}"
        d.rectangle([b[0], max(0, b[1] - 22), b[0] + 11 * len(tag), b[1]], fill=(255, 40, 40))
        d.text((b[0] + 2, max(0, b[1] - 22)), tag, fill=(255, 255, 255), font=font)
    d.text((6, 6), title, fill=(255, 255, 0), font=font)
    return im


def save_side_by_side(clean_panel, adv_panel, path):
    from PIL import Image
    w = clean_panel.width + adv_panel.width
    h = max(clean_panel.height, adv_panel.height)
    canvas = Image.new("RGB", (w, h), (20, 20, 20))
    canvas.paste(clean_panel, (0, 0))
    canvas.paste(adv_panel, (clean_panel.width, 0))
    canvas.save(path)


# --------------------------------------------------------------------------- #
# COCO mAP@50 (context metric) for a set of detections
# --------------------------------------------------------------------------- #
def coco_map50(coco_gt, results, img_ids):
    import contextlib
    import io
    from pycocotools.cocoeval import COCOeval
    if not results:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1])   # AP@0.50


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs_tt100k/yolov8s/weights/best.pt",
                    help="trained YOLOv8s TT100K checkpoint")
    ap.add_argument("--coco-val", default="coco/annotations/instances_val.json")
    ap.add_argument("--img-root", default="tt100k_2021")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="match clean-eval conf")
    ap.add_argument("--iou", type=float, default=0.7, help="match clean-eval NMS iou")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-images", type=int, default=500,
                    help="fixed subset size for the full sweep (state in report)")
    ap.add_argument("--max-iter", type=int, default=20, help="PGD-20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="20 images, eps=8/255 only, save before/after panels")
    ap.add_argument("--out-json", default="disappearance_yolo.json")
    ap.add_argument("--out-dir", default="disappearance_yolo_out")
    args = ap.parse_args()

    # ----- STEP 0: environment / version report (also lives here for reproducibility) -----
    try:
        import torch
        import torch.nn as nn
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import ops
        import art
        from art.estimators.object_detection import PyTorchYolo
        from art.attacks.evasion import ProjectedGradientDescent
    except Exception as e:
        sys.exit(f"ENV ERROR (import): {type(e).__name__}: {e}")
    print(f"ART {art.__version__} | torch {torch.__version__} | "
          f"ultralytics {ultralytics.__version__}")

    if not os.path.isfile(args.weights):
        sys.exit(f"ERROR: trained checkpoint not found: {args.weights}\n"
                 f"Train YOLO first (runs_tt100k/yolov8s/weights/best.pt) or pass --weights.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    classes = json.load(open(args.classes))
    names = classes["names"]
    nc = classes["num_classes"]

    # eps sweep (in [0,1] units; A4)
    if args.smoke:
        eps_list = [("8/255", 8 / 255)]
        n_images = 20
    else:
        eps_list = [("2/255", 2 / 255), ("4/255", 4 / 255), ("8/255", 8 / 255)]
        n_images = args.num_images

    # ----- model + ART estimator -----
    yolo = YOLO(args.weights)
    det_model = yolo.model.to(device).float()
    Wrapper = build_wrapper(torch, nn)
    wrapper = Wrapper(det_model, nc).to(device)
    estimator = PyTorchYolo(
        model=wrapper,
        input_shape=(3, args.imgsz, args.imgsz),
        channels_first=True,
        clip_values=(0.0, 1.0),                 # A4
        attack_losses=("loss_total",),          # A2
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    # ----- image subset (same images across all eps; from COCO val for GT) -----
    from pycocotools.coco import COCO
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.coco_val)
    all_img_ids = sorted(coco_gt.getImgIds())
    rng = random.Random(args.seed)
    img_ids = sorted(rng.sample(all_img_ids, min(n_images, len(all_img_ids))))
    print(f"images: {len(img_ids)} (subset of {len(all_img_ids)} val images, seed={args.seed})")
    print(f"{'SMOKE: ' if args.smoke else ''}eps sweep: {[e[0] for e in eps_list]}, "
          f"PGD-{args.max_iter}, conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")

    from PIL import Image

    def load_image(img_id):
        info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(args.img_root, info["file_name"])
        im = Image.open(path).convert("RGB").resize((args.imgsz, args.imgsz), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0          # HWC [0,1]
        chw = np.transpose(arr, (2, 0, 1))                       # CHW
        return chw, info["width"], info["height"]

    # pre-load and pre-detect CLEAN once (clean dets are eps-independent)
    cache = {}
    for img_id in img_ids:
        chw, W, H = load_image(img_id)
        x = torch.from_numpy(chw[None]).to(device)
        cb, cs, cl = detect(det_model, x, args.conf, args.iou, nc, torch, ops)
        cache[img_id] = {"chw": chw, "W": W, "H": H, "cb": cb, "cs": cs, "cl": cl}

    def to_coco_results(img_id, boxes, scores, labels):
        """imgsz-space xyxy -> original-coord xywh COCO results (category_id = label)."""
        c = cache[img_id]
        sx, sy = c["W"] / args.imgsz, c["H"] / args.imgsz
        res = []
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            res.append({"image_id": int(img_id), "category_id": int(l),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)})
        return res

    summary = {}
    for eps_name, eps in eps_list:
        attack = ProjectedGradientDescent(
            estimator=estimator, norm=np.inf, eps=eps, eps_step=eps / 4,
            max_iter=args.max_iter, targeted=False, batch_size=1, verbose=False)

        tot_clean = tot_gone = tot_relabel = tot_persist = tot_adv = 0
        conf_drops = []
        clean_results, adv_results = [], []
        n_panels = 0

        for img_id in img_ids:
            c = cache[img_id]
            chw, cb, cs, cl = c["chw"], c["cb"], c["cs"], c["cl"]
            clean_results += to_coco_results(img_id, cb, cs, cl)
            if len(cb) == 0:
                continue   # nothing detected clean -> can't disappear (A: skip)

            # untargeted PGD: y = the clean detections (drive the model away from them)
            y = [{"boxes": cb.astype(np.float32), "labels": cl.astype(np.int64)}]
            x_np = chw[None].astype(np.float32)
            x_adv = attack.generate(x=x_np, y=y)

            xa = torch.from_numpy(x_adv).to(device)
            ab, as_, al = detect(det_model, xa, args.conf, args.iou, nc, torch, ops)
            adv_results += to_coco_results(img_id, ab, as_, al)

            gone, relab, persist, drops = disappearance_stats(cb, cl, cs, ab, al, as_)
            tot_clean += len(cb); tot_gone += gone; tot_relabel += relab
            tot_persist += persist; tot_adv += len(ab); conf_drops += drops

            if args.smoke and n_panels < 4:
                cp = draw(chw, cb, cl, cs, names, f"CLEAN ({len(cb)} dets)")
                apnl = draw(x_adv[0], ab, al, as_, names, f"ADV eps={eps_name} ({len(ab)} dets)")
                save_side_by_side(cp, apnl, os.path.join(args.out_dir, f"panel_{img_id}.png"))
                n_panels += 1

        dis_rate = tot_gone / tot_clean if tot_clean else 0.0
        try:
            clean_map = coco_map50(coco_gt, clean_results, img_ids)
            adv_map = coco_map50(coco_gt, adv_results, img_ids)
        except Exception as e:
            print(f"  mAP computation failed ({type(e).__name__}: {e}); reporting None")
            clean_map = adv_map = None

        summary[eps_name] = {
            "eps_value": eps,
            "disappearance_rate": dis_rate,
            "clean_mAP50": clean_map,
            "adv_mAP50": adv_map,
            "clean_det_count": tot_clean,
            "adv_det_count": tot_adv,
            "relabeled_count": tot_relabel,
            "gone_count": tot_gone,
            "persisted_count": tot_persist,
            "mean_conf_drop": float(np.mean(conf_drops)) if conf_drops else 0.0,
        }
        print(f"  [{eps_name}] disappearance={dis_rate:.3f}  "
              f"clean_dets={tot_clean}->adv_dets={tot_adv}  "
              f"gone={tot_gone} relabeled={tot_relabel} persisted={tot_persist}  "
              f"mAP50 {clean_map}->{adv_map}")

    # ----- outputs -----
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # strength curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [summary[e]["eps_value"] * 255 for e, _ in eps_list]
        ys = [summary[e]["disappearance_rate"] for e, _ in eps_list]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, "o-", lw=2)
        plt.xlabel("eps (x/255, L-inf)"); plt.ylabel("disappearance rate")
        plt.title("YOLOv8s TT100K — PGD disappearance vs eps")
        plt.grid(True, alpha=0.3); plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "disappearance_vs_eps.png"), dpi=120)
    except Exception as e:
        print(f"  plot skipped ({type(e).__name__}: {e})")

    # summary table
    print("\n" + "=" * 64)
    print(f"{'eps':>8} | {'disappearance':>13} | {'clean->adv dets':>18}")
    print("-" * 64)
    for eps_name, _ in eps_list:
        s = summary[eps_name]
        print(f"{eps_name:>8} | {s['disappearance_rate']:>13.3f} | "
              f"{str(s['clean_det_count'])+'->'+str(s['adv_det_count']):>18}")
    print("=" * 64)
    print(f"JSON -> {args.out_json} | panels/plot -> {args.out_dir}/")


if __name__ == "__main__":
    main()
