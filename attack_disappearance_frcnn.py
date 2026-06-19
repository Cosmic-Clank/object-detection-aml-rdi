#!/usr/bin/env python3
"""
Digital white-box object-DISAPPEARANCE attack on the trained Faster R-CNN
(ResNet-50 FPN) TT100K detector, via ART's PyTorchObjectDetector.

Mirror of attack_disappearance_yolo.py for cross-detector comparability: SAME
disappearance metric (gone/relabeled/persisted; rate = gone/clean_dets), SAME
eps sweep, SAME PSNR/SSIM/LPIPS perceptual metrics, SAME JSON schema and plot.
The shared metric/plot/drawing helpers are imported unchanged from the YOLO
script so both detectors are measured identically.

Citations:
  * ART (Nicolae et al., 2018, arXiv:1807.01069) — attack implementation toolkit.
  * PGD (Madry et al., 2018, arXiv:1706.06083) — the L-inf attack.
  * TOG (Chow et al., 2020, arXiv:2004.04320) — disappearance/vanishing concept.

----------------------------------------------------------------------------
ASSUMPTIONS / CONVENTIONS:
  F1. ART's PyTorchObjectDetector is the NATIVE torchvision-detector case (no
      is_ultralytics flag): the model returns the 4 losses in train mode and
      post-NMS dicts in eval mode, so ART's default attack_losses
      (loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg) apply.
  F2. Labels stay in FRCNN's 1..45 space (background=0) for BOTH clean and adv
      detections — the disappearance metric only needs internal consistency.
      For COCO mAP (GT is 0..44) we map category_id = label - 1.
  F3. Images are 2048x2048 (square) -> resized to 1280x1280 [0,1] CHW, the SAME
      preprocessing/space as the YOLO attack, so the two are directly comparable.
      The model's transform (min_size=1280,max_size=2048) leaves 1280x1280 as-is.
  F4. vanish = targeted PGD toward EMPTY detections: descends the detection loss
      toward "no objects" -> RPN objectness + classifier confidence drop -> boxes
      fall below threshold -> disappearance. eps is the primary imperceptibility
      budget (Madry 2018); PSNR/SSIM/LPIPS are SUPPLEMENTARY (PGD is additive).

NOTE: requires fasterrcnn_tt100k_best.pth. Run --smoke FIRST, inspect panels.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

# reuse the YOLO script's metric/plot/draw helpers UNCHANGED (single source)
from attack_disappearance_yolo import (
    iou_matrix, disappearance_stats, draw, save_side_by_side, coco_map50,
)


# --------------------------------------------------------------------------- #
# model + detection (torchvision FRCNN returns post-NMS dicts; no manual NMS)
# --------------------------------------------------------------------------- #
def build_model(torch, num_classes, min_size, max_size, ckpt_path, device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # same construction as training (weights=None: we load our trained state)
    model = fasterrcnn_resnet50_fpn(weights=None, min_size=min_size, max_size=max_size)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)  # 45 + bg
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    return model.to(device).eval()


def detect_frcnn(model, x, conf, torch):
    """x: [1,3,H,W] (or [3,H,W]) in [0,1]. Returns boxes[xyxy], scores, labels(1..45)."""
    model.eval()
    img = x[0] if x.dim() == 4 else x
    with torch.no_grad():
        out = model([img])[0]
    boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
    labels = out["labels"].detach().cpu().numpy().astype(np.int64)
    scores = out["scores"].detach().cpu().numpy().astype(np.float32)
    keep = scores >= conf
    return boxes[keep], scores[keep], labels[keep]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="fasterrcnn_tt100k_best.pth",
                    help="trained Faster R-CNN TT100K checkpoint")
    ap.add_argument("--coco-val", default="coco/annotations/instances_val.json")
    ap.add_argument("--img-root", default="tt100k_2021")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-size", type=int, default=1280)
    ap.add_argument("--max-size", type=int, default=2048)
    ap.add_argument("--conf", type=float, default=0.5,
                    help="FRCNN score threshold (its natural operating point)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-images", type=int, default=500,
                    help="fixed subset size; SAME seed/subset as the YOLO run")
    ap.add_argument("--max-iter", type=int, default=20, help="PGD-20")
    ap.add_argument("--eps-list", default="1,2,4,6,8,12,16",
                    help="comma-separated L-inf budgets as k/255 (full sweep only)")
    ap.add_argument("--seed", type=int, default=0, help="MUST match YOLO run (0) for identical images")
    ap.add_argument("--mode", choices=["vanish", "untargeted"], default="vanish",
                    help="vanish: targeted PGD toward NO objects (clean disappearance). "
                         "untargeted: maximize detection loss away from clean dets.")
    ap.add_argument("--smoke", action="store_true",
                    help="20 images, eps=8/255 only, save before/after panels")
    ap.add_argument("--out-json", default="disappearance_frcnn.json")
    ap.add_argument("--out-dir", default="disappearance_frcnn_out")
    args = ap.parse_args()

    # ----- STEP 0: env + imports -----
    try:
        import torch
        import torchvision
        import art
        from art.estimators.object_detection import PyTorchObjectDetector
        from art.attacks.evasion import ProjectedGradientDescent
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn
        from pycocotools.coco import COCO
        from PIL import Image
    except Exception as e:
        sys.exit(f"ENV ERROR (import): {type(e).__name__}: {e}")
    print(f"ART {art.__version__} | torch {torch.__version__} | "
          f"torchvision {torchvision.__version__}")

    if not os.path.isfile(args.weights):
        sys.exit(f"ERROR: FRCNN checkpoint not found: {args.weights}\n"
                 f"Train Faster R-CNN first (fasterrcnn_tt100k_best.pth) or pass --weights.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    classes = json.load(open(args.classes))
    names = classes["names"]                       # id 0..44 -> name
    nc = classes["num_classes"]
    num_classes = nc + 1                            # + background
    names_draw = ["__bg__"] + names                 # index by FRCNN label (1..45)

    if args.smoke:
        eps_list = [("8/255", 8 / 255)]
        n_images = 20
    else:
        eps_list = []
        for tok in args.eps_list.split(","):
            tok = tok.strip()
            if tok:
                k = float(tok)
                eps_list.append((f"{int(k) if k == int(k) else k}/255", k / 255.0))
        n_images = args.num_images

    # ----- model + ART estimator (native torchvision path) -----
    model = build_model(torch, num_classes, args.min_size, args.max_size, args.weights, device)
    estimator = PyTorchObjectDetector(
        model=model,
        input_shape=(3, args.imgsz, args.imgsz),
        clip_values=(0.0, 1.0),
        channels_first=True,
        attack_losses=("loss_classifier", "loss_box_reg",
                       "loss_objectness", "loss_rpn_box_reg"),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    # ----- image subset: SAME seed + instances_val.json as YOLO -> same images -----
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.coco_val)
    all_img_ids = sorted(coco_gt.getImgIds())
    rng = random.Random(args.seed)
    img_ids = sorted(rng.sample(all_img_ids, min(n_images, len(all_img_ids))))
    print(f"images: {len(img_ids)} (subset of {len(all_img_ids)} val images, seed={args.seed})")
    print(f"{'SMOKE: ' if args.smoke else ''}mode={args.mode}, eps={[e[0] for e in eps_list]}, "
          f"PGD-{args.max_iter}, conf={args.conf}, imgsz={args.imgsz}")

    # ----- imperceptibility (additive metrics; eps is primary, these supplementary) -----
    lpips_model = None
    try:
        import lpips as _lpips
        lpips_model = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        print("imperceptibility: PSNR + SSIM (skimage) + LPIPS(alex) ready")
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

    def load_image(img_id):
        info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(args.img_root, info["file_name"])
        im = Image.open(path).convert("RGB").resize((args.imgsz, args.imgsz), Image.BILINEAR)
        chw = np.transpose(np.asarray(im, np.float32) / 255.0, (2, 0, 1))
        return chw, info["width"], info["height"]

    cache = {}
    for img_id in img_ids:
        chw, W, H = load_image(img_id)
        x = torch.from_numpy(chw[None]).to(device)
        cb, cs, cl = detect_frcnn(model, x, args.conf, torch)
        cache[img_id] = {"chw": chw, "W": W, "H": H, "cb": cb, "cs": cs, "cl": cl}

    def to_coco_results(img_id, boxes, scores, labels):
        """imgsz-space xyxy -> original-coord xywh; category_id = label-1 (1..45 -> 0..44)."""
        c = cache[img_id]
        sx, sy = c["W"] / args.imgsz, c["H"] / args.imgsz
        res = []
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            res.append({"image_id": int(img_id), "category_id": int(l) - 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)})
        return res

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
                continue                            # nothing detected clean -> skip

            if args.mode == "vanish":
                y = [{"boxes": np.zeros((0, 4), np.float32),
                      "labels": np.zeros((0,), np.int64)}]
            else:
                y = [{"boxes": cb.astype(np.float32), "labels": cl.astype(np.int64)}]
            x_np = chw[None].astype(np.float32)
            x_adv = attack.generate(x=x_np, y=y)

            xa = torch.from_numpy(x_adv).to(device)
            ab, as_, al = detect_frcnn(model, xa, args.conf, torch)
            adv_results += to_coco_results(img_id, ab, as_, al)

            gone, relab, persist, drops = disappearance_stats(cb, cl, cs, ab, al, as_)
            tot_clean += len(cb); tot_gone += gone; tot_relabel += relab
            tot_persist += persist; tot_adv += len(ab); conf_drops += drops

            p, s, lp = perceptual(chw, x_adv[0])
            psnrs.append(p); ssims.append(s)
            if lp is not None:
                lpipss.append(lp)

            if args.smoke and n_panels < 4:
                cp = draw(chw, cb, cl, cs, names_draw, f"CLEAN ({len(cb)} dets)")
                apnl = draw(x_adv[0], ab, al, as_, names_draw, f"ADV eps={eps_name} ({len(ab)} dets)")
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

        fig.suptitle("FasterRCNN-R50 TT100K — PGD disappearance: stealth-vs-potency")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "disappearance_vs_eps.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped ({type(e).__name__}: {e})")

    print("\n" + "=" * 92)
    print(f"{'eps':>7} | {'disappear':>9} | {'clean->adv':>12} | {'gone/relab':>11} | "
          f"{'PSNR(dB)':>8} | {'SSIM':>6} | {'LPIPS':>6}")
    print("-" * 92)
    for eps_name, _ in eps_list:
        s = summary[eps_name]
        lp = f"{s['lpips_mean']:.4f}" if s["lpips_mean"] is not None else "n/a"
        print(f"{eps_name:>7} | {s['disappearance_rate']:>9.3f} | "
              f"{str(s['clean_det_count'])+'->'+str(s['adv_det_count']):>12} | "
              f"{str(s['gone_count'])+'/'+str(s['relabeled_count']):>11} | "
              f"{s['psnr_mean']:>8.2f} | {s['ssim_mean']:>6.4f} | {lp:>6}")
    print("=" * 92)
    print(f"FasterRCNN-R50 TT100K disappearance ({args.mode}) | JSON -> {args.out_json} "
          f"| panels/plot -> {args.out_dir}/")


if __name__ == "__main__":
    main()
