#!/usr/bin/env python3
"""
Digital white-box TARGETED MISCLASSIFICATION evasion attack on the trained TT100K
detectors (YOLOv8s or Faster R-CNN), via ART. INFERENCE-TIME evasion: the model
is clean; we perturb the input (NOT poisoning/training).

Mirrors attack_disappearance_{yolo,frcnn}.py: same ART (PyTorchYolo /
PyTorchObjectDetector), same PGD, same eps sweep {1,2,4,6,8,12,16}/255, same
500-image subset + seed=0 + instances_val.json, same PSNR/SSIM/LPIPS perceptual
metrics. Shared metric/plot/draw + per-detector model/detect helpers are imported
from the disappearance scripts so all rows are directly comparable.

CONTRAST WITH DISAPPEARANCE: disappearance drove objectness/confidence -> 0 (boxes
vanish). Here we run TARGETED PGD that KEEPS the source box but flips its label to
an attacker-chosen target class (gradient from the CLASSIFICATION loss toward the
target). Box persists, label changes.

CHOSEN PAIR: pl40 -> pl80 (40 km/h sign misclassified as 80 km/h).
  Why: both are top speed-limit classes by instance count (pl40: 901 train/455 val;
  pl80: 587/278) -> strongly detected by both detectors (not confounded by an
  uncertain class; baseline val mAP50=0.915). pl40 has the most val instances of
  any speed limit -> stable ASR over the 500-image subset. Threat: doubling the
  perceived limit (40->80) is a dangerous smart-city overspeed scenario.

Citations:
  * ART (Nicolae et al., 2018, arXiv:1807.01069) — attack toolkit.
  * PGD (Madry et al., 2018, arXiv:1706.06083) — the L-inf attack.
  * TOG (Chow et al., 2020, arXiv:2004.04320) — targeted object-mislabeling concept.

TARGETED SUCCESS METRIC (per source-class object detected on the clean image):
  TARGETED_SUCCESS = adv detection matches its box (IoU>=0.5) AND label==target.
  OTHER_WRONG      = matched box, label != source and != target (partial).
  STILL_SOURCE     = matched box, still source label (attack failed).
  VANISHED         = no matching adv box (attack disappeared it; NOT a success).
  targeted_success_rate = TARGETED_SUCCESS / (source objects detected clean).

NOTE: needs the trained checkpoint for the chosen --detector. Run --smoke first.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

# shared helpers, imported UNCHANGED from the disappearance scripts
from attack_disappearance_yolo import iou_matrix, draw, save_side_by_side, coco_map50, detect
from attack_disappearance_frcnn import build_model, detect_frcnn


# --------------------------------------------------------------------------- #
# targeted-misclassification bookkeeping for one image
# --------------------------------------------------------------------------- #
def misclass_stats(src_boxes, adv_boxes, adv_labels, source_id, target_id, iou_thr=0.5):
    """For each clean source-class box, classify its fate on the adv image.
    Returns (targeted_success, other_wrong, still_source, vanished)."""
    n = len(src_boxes)
    tgt = other = still = vanished = 0
    if n == 0:
        return 0, 0, 0, 0
    ious = iou_matrix(src_boxes, adv_boxes)          # [n, m]
    for i in range(n):
        if ious.shape[1] == 0 or ious[i].max() < iou_thr:
            vanished += 1
            continue
        lab = int(adv_labels[int(np.argmax(ious[i]))])
        if lab == target_id:
            tgt += 1
        elif lab == source_id:
            still += 1
        else:
            other += 1
    return tgt, other, still, vanished


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["yolo", "frcnn"], required=True)
    ap.add_argument("--weights", default=None,
                    help="checkpoint; default per detector "
                         "(yolo: runs_tt100k/yolov8s/weights/best.pt, frcnn: fasterrcnn_tt100k_best.pth)")
    ap.add_argument("--source", default="pl40")
    ap.add_argument("--target", default="pl80")
    ap.add_argument("--coco-val", default="coco/annotations/instances_val.json")
    ap.add_argument("--img-root", default="tt100k_2021")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-size", type=int, default=1280, help="frcnn transform")
    ap.add_argument("--max-size", type=int, default=2048, help="frcnn transform")
    ap.add_argument("--conf", type=float, default=None,
                    help="score threshold (default yolo 0.25, frcnn 0.5)")
    ap.add_argument("--iou", type=float, default=0.7, help="yolo NMS iou")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-images", type=int, default=500, help="same subset as disappearance")
    ap.add_argument("--max-iter", type=int, default=20, help="PGD-20")
    ap.add_argument("--eps-list", default="1,2,4,6,8,12,16")
    ap.add_argument("--seed", type=int, default=0, help="MUST be 0 to match other runs")
    ap.add_argument("--model-name", default="yolov8s", help="ART is_ultralytics loss selector")
    ap.add_argument("--smoke", action="store_true",
                    help="20 images containing source signs, eps=8/255, save panels")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    det = args.detector
    if args.weights is None:
        args.weights = ("runs_tt100k/yolov8s/weights/best.pt" if det == "yolo"
                        else "fasterrcnn_tt100k_best.pth")
    if args.conf is None:
        args.conf = 0.25 if det == "yolo" else 0.5
    if args.out_json is None:
        args.out_json = f"misclassification_{det}.json"
    if args.out_dir is None:
        args.out_dir = f"misclassification_{det}_out"

    # ----- STEP 0: env + imports -----
    try:
        import torch
        import art
        from art.attacks.evasion import ProjectedGradientDescent
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn
        from pycocotools.coco import COCO
        from PIL import Image
        if det == "yolo":
            import ultralytics
            from ultralytics import YOLO
            from art.estimators.object_detection import PyTorchYolo
            try:
                from ultralytics.utils.nms import non_max_suppression as nms_fn
            except ImportError:
                from ultralytics.utils.ops import non_max_suppression as nms_fn
            backend = f"ultralytics {ultralytics.__version__}"
        else:
            import torchvision
            from art.estimators.object_detection import PyTorchObjectDetector
            backend = f"torchvision {torchvision.__version__}"
    except Exception as e:
        sys.exit(f"ENV ERROR (import): {type(e).__name__}: {e}")
    print(f"ART {art.__version__} | torch {torch.__version__} | {backend} | detector={det}")

    if not os.path.isfile(args.weights):
        sys.exit(f"ERROR: checkpoint not found: {args.weights} (train {det} first or pass --weights)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    classes = json.load(open(args.classes))
    names = classes["names"]
    n2i = classes["name_to_id"]
    nc = classes["num_classes"]
    if args.source not in n2i or args.target not in n2i:
        sys.exit(f"ERROR: source/target must be in classes.json names")

    # label spaces: yolo 0..44 ; frcnn 1..45 (background=0). category_id (COCO GT) = base id.
    base_src, base_tgt = n2i[args.source], n2i[args.target]
    off = 0 if det == "yolo" else 1
    source_id, target_id = base_src + off, base_tgt + off
    names_draw = names if det == "yolo" else (["__bg__"] + names)
    cat_off = 0 if det == "yolo" else -1                    # label -> COCO category_id

    # eps sweep
    if args.smoke:
        eps_list = [("8/255", 8 / 255)]
    else:
        eps_list = []
        for tok in args.eps_list.split(","):
            tok = tok.strip()
            if tok:
                k = float(tok)
                eps_list.append((f"{int(k) if k == int(k) else k}/255", k / 255.0))

    # ----- model + ART estimator (per detector) -----
    if det == "yolo":
        yolo = YOLO(args.weights)
        det_model = yolo.model.to(device).float()
        estimator = PyTorchYolo(
            model=det_model, is_ultralytics=True, model_name=args.model_name,
            input_shape=(3, args.imgsz, args.imgsz), channels_first=True,
            clip_values=(0.0, 1.0), attack_losses=("loss_total",),
            device_type="gpu" if device.type == "cuda" else "cpu")

        def detect_fn(x):
            return detect(det_model, x, args.conf, args.iou, nc, torch, nms_fn)
    else:
        det_model = build_model(torch, nc + 1, args.min_size, args.max_size, args.weights, device)
        estimator = PyTorchObjectDetector(
            model=det_model, input_shape=(3, args.imgsz, args.imgsz),
            clip_values=(0.0, 1.0), channels_first=True,
            attack_losses=("loss_classifier", "loss_box_reg",
                           "loss_objectness", "loss_rpn_box_reg"),
            device_type="gpu" if device.type == "cuda" else "cpu")

        def detect_fn(x):
            return detect_frcnn(det_model, x, args.conf, torch)

    # ----- perceptual metrics (additive; eps is primary, these supplementary) -----
    lpips_model = None
    try:
        import lpips as _lpips
        lpips_model = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        print("imperceptibility: PSNR + SSIM + LPIPS(alex) ready")
    except Exception as e:
        print(f"imperceptibility: LPIPS unavailable ({type(e).__name__}: {e}); PSNR + SSIM only")

    def perceptual(clean_chw, adv_chw):
        p = float(psnr_fn(clean_chw, adv_chw, data_range=1.0))
        s = float(ssim_fn(clean_chw, adv_chw, channel_axis=0, data_range=1.0))
        lp = None
        if lpips_model is not None:
            with torch.no_grad():
                a = torch.from_numpy(clean_chw[None].astype(np.float32) * 2 - 1).to(device)
                b = torch.from_numpy(adv_chw[None].astype(np.float32) * 2 - 1).to(device)
                lp = float(lpips_model(a, b))
        return p, s, lp

    # ----- image set -----
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.coco_val)
    all_img_ids = sorted(coco_gt.getImgIds())

    def load_image(img_id):
        info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(args.img_root, info["file_name"])
        im = Image.open(path).convert("RGB").resize((args.imgsz, args.imgsz), Image.BILINEAR)
        chw = np.transpose(np.asarray(im, np.float32) / 255.0, (2, 0, 1))
        return chw, info["width"], info["height"]

    # cache CLEAN detections. smoke: scan until 20 images contain a source sign.
    # full: SAME 500-subset (seed 0) as disappearance for comparability.
    rng = random.Random(args.seed)
    cache, img_ids = {}, []
    if args.smoke:
        for img_id in rng.sample(all_img_ids, len(all_img_ids)):
            chw, W, H = load_image(img_id)
            cb, cs, cl = detect_fn(torch.from_numpy(chw[None]).to(device))
            if (cl == source_id).any():
                cache[img_id] = {"chw": chw, "W": W, "H": H, "cb": cb, "cs": cs, "cl": cl}
                img_ids.append(img_id)
                if len(img_ids) >= 20:
                    break
    else:
        img_ids = sorted(rng.sample(all_img_ids, min(args.num_images, len(all_img_ids))))
        for img_id in img_ids:
            chw, W, H = load_image(img_id)
            cb, cs, cl = detect_fn(torch.from_numpy(chw[None]).to(device))
            cache[img_id] = {"chw": chw, "W": W, "H": H, "cb": cb, "cs": cs, "cl": cl}

    src_total = sum(int((cache[i]["cl"] == source_id).sum()) for i in img_ids)
    print(f"detector={det} | pair {args.source}(id {source_id}) -> {args.target}(id {target_id})")
    print(f"images: {len(img_ids)} (seed={args.seed}{'' if args.smoke else ', same 500-subset as disappearance'})")
    print(f"source '{args.source}' objects detected clean in eval set: {src_total}")
    if src_total < 30:
        print(f"  WARNING: only {src_total} source objects — ASR may be noisy; consider --num-images larger")
    print(f"{'SMOKE: ' if args.smoke else ''}eps={[e[0] for e in eps_list]}, PGD-{args.max_iter}, conf={args.conf}")

    justification = (f"{args.source}->{args.target}: both top speed-limit classes by instance count "
                     f"(strongly detected by both detectors); {args.source} most frequent speed limit "
                     f"in val -> stable ASR; doubling perceived limit is a dangerous overspeed scenario.")

    def to_coco_results(img_id, boxes, scores, labels):
        c = cache[img_id]; sx, sy = c["W"] / args.imgsz, c["H"] / args.imgsz
        return [{"image_id": int(img_id), "category_id": int(l) + cat_off,
                 "bbox": [float(b[0] * sx), float(b[1] * sy),
                          float((b[2] - b[0]) * sx), float((b[3] - b[1]) * sy)],
                 "score": float(s)} for b, s, l in zip(boxes, scores, labels)]

    summary = {"detector": det, "source": args.source, "target": args.target,
               "source_id": source_id, "target_id": target_id,
               "source_objects_eval": src_total, "justification": justification, "eps": {}}

    for eps_name, eps in eps_list:
        attack = ProjectedGradientDescent(
            estimator=estimator, norm=np.inf, eps=eps, eps_step=eps / 4,
            max_iter=args.max_iter, targeted=True, batch_size=1, verbose=False)

        tot_src = tgt_s = oth_w = still_s = van = 0
        psnrs, ssims, lpipss = [], [], []
        n_panels = 0

        for img_id in img_ids:
            c = cache[img_id]
            chw, cb, cs, cl = c["chw"], c["cb"], c["cs"], c["cl"]
            src_mask = (cl == source_id)
            if not src_mask.any():
                continue                                     # no source object here
            # targeted y: KEEP all clean detections, relabel only source -> target
            tgt_labels = cl.copy()
            tgt_labels[src_mask] = target_id
            y = [{"boxes": cb.astype(np.float32), "labels": tgt_labels.astype(np.int64)}]
            x_adv = attack.generate(x=chw[None].astype(np.float32), y=y)

            ab, as_, al = detect_fn(torch.from_numpy(x_adv).to(device))
            t, o, st, v = misclass_stats(cb[src_mask], ab, al, source_id, target_id)
            n_src = int(src_mask.sum())
            tot_src += n_src; tgt_s += t; oth_w += o; still_s += st; van += v

            p, s, lp = perceptual(chw, x_adv[0])
            psnrs.append(p); ssims.append(s)
            if lp is not None:
                lpipss.append(lp)

            if args.smoke and n_panels < 4:
                cp = draw(chw, cb[src_mask], cl[src_mask], cs[src_mask], names_draw,
                          f"CLEAN {args.source} ({n_src})")
                apnl = draw(x_adv[0], ab, al, as_, names_draw, f"ADV eps={eps_name} ->{args.target}")
                save_side_by_side(cp, apnl, os.path.join(args.out_dir, f"panel_{img_id}.png"))
                n_panels += 1

        asr = tgt_s / tot_src if tot_src else 0.0
        e = {
            "eps_value": eps, "targeted_success_rate": asr,
            "source_objects": tot_src, "targeted_success": tgt_s, "other_wrong": oth_w,
            "still_source": still_s, "vanished": van,
            "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
            "ssim_mean": float(np.mean(ssims)) if ssims else None,
            "lpips_mean": float(np.mean(lpipss)) if lpipss else None,
            "perceptual_count": len(psnrs),
        }
        summary["eps"][eps_name] = e
        lp_str = f"{e['lpips_mean']:.4f}" if e["lpips_mean"] is not None else "n/a"
        print(f"  [{eps_name}] ASR(targeted)={asr:.3f}  src={tot_src}  "
              f"targeted={tgt_s} other_wrong={oth_w} still_source={still_s} vanished={van}  "
              f"PSNR={e['psnr_mean']:.2f}dB SSIM={e['ssim_mean']:.4f} LPIPS={lp_str}")

    # ----- outputs -----
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [summary["eps"][e]["eps_value"] * 255 for e, _ in eps_list]
        asrs = [summary["eps"][e]["targeted_success_rate"] for e, _ in eps_list]
        psnr_v = [summary["eps"][e]["psnr_mean"] for e, _ in eps_list]
        ssim_v = [summary["eps"][e]["ssim_mean"] for e, _ in eps_list]
        lpips_v = [summary["eps"][e]["lpips_mean"] for e, _ in eps_list]
        have_lp = all(v is not None for v in lpips_v)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
        ax0.plot(xs, asrs, "o-", color="crimson", lw=2)
        ax0.set_xlabel("eps (x/255, L-inf)"); ax0.set_ylabel("targeted success rate")
        ax0.set_ylim(0, 1); ax0.grid(True, alpha=0.3)
        ax0.set_title(f"potency: {args.source}->{args.target} ASR vs eps")

        ax1.plot(xs, ssim_v, "o-", color="tab:blue", label="SSIM")
        if have_lp:
            ax1.plot(xs, lpips_v, "s-", color="tab:green", label="LPIPS")
        ax1.set_xlabel("eps (x/255, L-inf)"); ax1.set_ylabel("SSIM / LPIPS")
        ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3)
        axr = ax1.twinx(); axr.plot(xs, psnr_v, "^--", color="tab:orange", label="PSNR (dB)")
        axr.set_ylabel("PSNR (dB)")
        ax1.set_title("stealth: imperceptibility vs eps")
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        fig.suptitle(f"{det.upper()} TT100K — targeted misclassification {args.source}->{args.target}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "misclassification_vs_eps.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped ({type(e).__name__}: {e})")

    print("\n" + "=" * 94)
    print(f"{'eps':>7} | {'ASR':>6} | {'targeted':>8} | {'oth_wrong':>9} | {'still_src':>9} | "
          f"{'vanished':>8} | {'PSNR':>6} | {'SSIM':>6} | {'LPIPS':>6}")
    print("-" * 94)
    for eps_name, _ in eps_list:
        s = summary["eps"][eps_name]
        lp = f"{s['lpips_mean']:.3f}" if s["lpips_mean"] is not None else "n/a"
        print(f"{eps_name:>7} | {s['targeted_success_rate']:>6.3f} | {s['targeted_success']:>8} | "
              f"{s['other_wrong']:>9} | {s['still_source']:>9} | {s['vanished']:>8} | "
              f"{s['psnr_mean']:>6.2f} | {s['ssim_mean']:>6.4f} | {lp:>6}")
    print("=" * 94)
    print(f"{det.upper()} {args.source}->{args.target} | JSON -> {args.out_json} | panels/plot -> {args.out_dir}/")


if __name__ == "__main__":
    main()
