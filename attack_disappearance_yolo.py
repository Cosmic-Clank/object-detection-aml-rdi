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
  A1. We use ART >= 1.20's native Ultralytics path (is_ultralytics=True,
      model_name=...): ART wraps the DetectionModel with PyTorchYoloLossWrapper
      (sets v8DetectionLoss). For the disappearance METRIC we still run
      Ultralytics' own NMS in detect() (controls conf/iou, matches clean eval)
      and pass y explicitly to PGD, so the metric never depends on ART.predict.
  A2. Untargeted PGD maximizes Ultralytics' v8 detection loss (box+cls+dfl) w.r.t.
      the clean detections passed as y. YOLOv8 is anchor-free with no objectness
      channel, so the cls term drives class confidence below the conf threshold
      -> the box drops out at NMS -> disappearance. ART's wrapper exposes
      'loss_total'; we set attack_losses=('loss_total',).
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
# detection via Ultralytics NMS (identical path for clean and adversarial)
# --------------------------------------------------------------------------- #
def detect(det_model, x, conf, iou, nc, torch, nms_fn):
    """x: [1,3,H,W] in [0,1]. Returns numpy boxes[n,4] xyxy, scores[n], labels[n]."""
    det_model.eval()
    with torch.no_grad():
        preds = det_model(x)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        out = nms_fn(preds, conf_thres=conf, iou_thres=iou, nc=nc)[0]
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
    W, H = im.size
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for b, l, s in zip(boxes, labels, scores):
        # clamp to image bounds and order coords; adv NMS can yield out-of-bounds
        # or inverted boxes, which PIL.rectangle rejects (y1 >= y0 required).
        x0 = min(max(float(b[0]), 0), W - 1)
        y0 = min(max(float(b[1]), 0), H - 1)
        x1 = min(max(float(b[2]), 0), W - 1)
        y1 = min(max(float(b[3]), 0), H - 1)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=3)
        tag = f"{names[int(l)]} {s:.2f}"
        ty0 = max(0.0, y0 - 22)
        ty1 = max(ty0, y0)                       # guarantee ty1 >= ty0
        d.rectangle([x0, ty0, min(x0 + 11 * len(tag), W - 1), ty1], fill=(255, 40, 40))
        d.text((x0 + 2, ty0), tag, fill=(255, 255, 255), font=font)
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
    ap.add_argument("--eps-list", default="1,2,4,6,8,12,16",
                    help="comma-separated L-inf budgets as k/255 (full sweep only); "
                         "default 1,2,4,6,8,12,16 -> {1..16}/255")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="20 images, eps=8/255 only, save before/after panels")
    ap.add_argument("--out-json", default="disappearance_yolo.json")
    ap.add_argument("--out-dir", default="disappearance_yolo_out")
    ap.add_argument("--model-name", default="yolov8s",
                    help="ART loss selector for is_ultralytics (v8 vs v10)")
    ap.add_argument("--mode", choices=["vanish", "untargeted"], default="vanish",
                    help="vanish: targeted PGD toward NO objects (minimize all class "
                         "confidence; TOG-style clean disappearance). untargeted: "
                         "maximize detection loss away from clean dets (v5-style; on "
                         "v8 this FABRICATES false positives instead of vanishing).")
    args = ap.parse_args()

    # ----- STEP 0: environment / version report (also lives here for reproducibility) -----
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        # non_max_suppression moved from ultralytics.utils.ops -> .nms in 8.4.x;
        # try the new location first, fall back for older versions.
        try:
            from ultralytics.utils.nms import non_max_suppression as nms_fn
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression as nms_fn
        import art
        from art.estimators.object_detection import PyTorchYolo
        from art.attacks.evasion import ProjectedGradientDescent
        # imperceptibility metrics (skimage required; lpips optional, set up below)
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn
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

    # eps sweep (in [0,1] units; A4). Each entry is parsed as k/255.
    def parse_eps(spec):
        out = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            k = float(tok)
            name = f"{int(k) if k == int(k) else k}/255"
            out.append((name, k / 255.0))
        return out

    if args.smoke:
        eps_list = [("8/255", 8 / 255)]
        n_images = 20
    else:
        eps_list = parse_eps(args.eps_list)
        n_images = args.num_images

    # ----- model + ART estimator (native Ultralytics path, ART >= 1.20) -----
    # ART wraps the DetectionModel with its PyTorchYoloLossWrapper, which sets up
    # v8DetectionLoss and returns {loss_total, loss_box, loss_cls, loss_dfl}.
    # We still run our own NMS in detect() for the metric (controls conf/iou and
    # avoids depending on ART's predict). (A1, A2)
    yolo = YOLO(args.weights)
    det_model = yolo.model.to(device).float()
    estimator = PyTorchYolo(
        model=det_model,
        is_ultralytics=True,
        model_name=args.model_name,
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
        cb, cs, cl = detect(det_model, x, args.conf, args.iou, nc, torch, nms_fn)
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

    # ----- imperceptibility metrics (ADDITIVE measurement; attack unchanged) -----
    # Why these apply here: PGD is an ADDITIVE L-inf perturbation (unlike a
    # geometric warp such as WaNet), so PSNR/SSIM/LPIPS -- which are additive-
    # perturbation similarity metrics -- are the correct tools. The PRIMARY
    # imperceptibility measure is still eps itself (the L-inf budget / standard
    # PGD threat model, Madry et al. 2018); PSNR/SSIM/LPIPS are SUPPLEMENTARY.
    # LPIPS model is instantiated ONCE here (loads a network), not per image.
    lpips_model = None
    try:
        import lpips as _lpips
        lpips_model = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        print("imperceptibility: PSNR + SSIM (skimage) + LPIPS(alex) ready")
    except Exception as e:
        print(f"imperceptibility: LPIPS unavailable ({type(e).__name__}: {e}); "
              f"reporting PSNR + SSIM only")

    def perceptual(clean_chw, adv_chw):
        """clean/adv: [0,1] CHW float (same space fed to the attack)."""
        p = float(psnr_fn(clean_chw, adv_chw, data_range=1.0))
        s = float(ssim_fn(clean_chw, adv_chw, channel_axis=0, data_range=1.0))
        lp = None
        if lpips_model is not None:
            with torch.no_grad():                 # LPIPS expects inputs in [-1,1]
                a = torch.from_numpy(clean_chw[None].astype(np.float32) * 2 - 1).to(device)
                b = torch.from_numpy(adv_chw[None].astype(np.float32) * 2 - 1).to(device)
                lp = float(lpips_model(a, b))
        return p, s, lp

    summary = {}
    for eps_name, eps in eps_list:
        attack = ProjectedGradientDescent(
            estimator=estimator, norm=np.inf, eps=eps, eps_step=eps / 4,
            max_iter=args.max_iter, targeted=(args.mode == "vanish"),
            batch_size=1, verbose=False)

        tot_clean = tot_gone = tot_relabel = tot_persist = tot_adv = 0
        conf_drops = []
        psnrs, ssims, lpipss = [], [], []
        clean_results, adv_results = [], []
        n_panels = 0

        for img_id in img_ids:
            c = cache[img_id]
            chw, cb, cs, cl = c["chw"], c["cb"], c["cs"], c["cl"]
            clean_results += to_coco_results(img_id, cb, cs, cl)
            if len(cb) == 0:
                continue   # nothing detected clean -> can't disappear (A: skip)

            if args.mode == "vanish":
                # targeted toward NO objects: descend the all-background cls loss
                # -> push every detection below conf threshold (clean disappearance).
                y = [{"boxes": np.zeros((0, 4), np.float32),
                      "labels": np.zeros((0,), np.int64)}]
            else:
                # untargeted: drive the model away from its own clean detections.
                y = [{"boxes": cb.astype(np.float32), "labels": cl.astype(np.int64)}]
            x_np = chw[None].astype(np.float32)
            x_adv = attack.generate(x=x_np, y=y)

            xa = torch.from_numpy(x_adv).to(device)
            ab, as_, al = detect(det_model, xa, args.conf, args.iou, nc, torch, nms_fn)
            adv_results += to_coco_results(img_id, ab, as_, al)

            gone, relab, persist, drops = disappearance_stats(cb, cl, cs, ab, al, as_)
            tot_clean += len(cb); tot_gone += gone; tot_relabel += relab
            tot_persist += persist; tot_adv += len(ab); conf_drops += drops

            # imperceptibility on the clean-vs-adv pair (additive measurement only)
            p, s, lp = perceptual(chw, x_adv[0])
            psnrs.append(p); ssims.append(s)
            if lp is not None:
                lpipss.append(lp)

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
            "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
            "ssim_mean": float(np.mean(ssims)) if ssims else None,
            "lpips_mean": float(np.mean(lpipss)) if lpipss else None,
            "perceptual_count": len(psnrs),
        }
        s_ = summary[eps_name]
        lp_str = f"{s_['lpips_mean']:.4f}" if s_["lpips_mean"] is not None else "n/a"
        print(f"  [{eps_name}] disappearance={dis_rate:.3f}  "
              f"clean_dets={tot_clean}->adv_dets={tot_adv}  "
              f"gone={tot_gone} relabeled={tot_relabel} persisted={tot_persist}  "
              f"mAP50 {clean_map}->{adv_map}  "
              f"PSNR={s_['psnr_mean']:.2f}dB SSIM={s_['ssim_mean']:.4f} LPIPS={lp_str}")

    # ----- outputs -----
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # stealth-vs-potency curve: disappearance rises with eps (potency) while
    # PSNR/SSIM fall and LPIPS rises (stealth degrades).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [summary[e]["eps_value"] * 255 for e, _ in eps_list]
        dis = [summary[e]["disappearance_rate"] for e, _ in eps_list]
        psnr_v = [summary[e]["psnr_mean"] for e, _ in eps_list]
        ssim_v = [summary[e]["ssim_mean"] for e, _ in eps_list]
        lpips_v = [summary[e]["lpips_mean"] for e, _ in eps_list]
        have_lp = all(v is not None for v in lpips_v)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
        ax0.plot(xs, dis, "o-", color="crimson", lw=2)
        ax0.set_xlabel("eps (x/255, L-inf)"); ax0.set_ylabel("disappearance rate")
        ax0.set_ylim(0, 1); ax0.grid(True, alpha=0.3)
        ax0.set_title("potency: disappearance vs eps")

        ax1.plot(xs, ssim_v, "o-", color="tab:blue", label="SSIM")
        if have_lp:
            ax1.plot(xs, lpips_v, "s-", color="tab:green", label="LPIPS")
        ax1.set_xlabel("eps (x/255, L-inf)"); ax1.set_ylabel("SSIM / LPIPS")
        ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3)
        axr = ax1.twinx()
        axr.plot(xs, psnr_v, "^--", color="tab:orange", label="PSNR (dB)")
        axr.set_ylabel("PSNR (dB)")
        ax1.set_title("stealth: imperceptibility vs eps")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

        fig.suptitle("YOLOv8s TT100K — PGD disappearance: stealth-vs-potency")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "disappearance_vs_eps.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped ({type(e).__name__}: {e})")

    # summary table
    print("\n" + "=" * 86)
    print(f"{'eps':>7} | {'disappear':>9} | {'clean->adv':>12} | {'PSNR(dB)':>8} | "
          f"{'SSIM':>6} | {'LPIPS':>6}")
    print("-" * 86)
    for eps_name, _ in eps_list:
        s = summary[eps_name]
        lp = f"{s['lpips_mean']:.4f}" if s["lpips_mean"] is not None else "n/a"
        print(f"{eps_name:>7} | {s['disappearance_rate']:>9.3f} | "
              f"{str(s['clean_det_count'])+'->'+str(s['adv_det_count']):>12} | "
              f"{s['psnr_mean']:>8.2f} | {s['ssim_mean']:>6.4f} | {lp:>6}")
    print("=" * 86)
    print(f"JSON -> {args.out_json} | panels/plot -> {args.out_dir}/")


if __name__ == "__main__":
    main()
