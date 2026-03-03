"""
Evaluate saved Sponges predictions against ground-truth label masks.

Usage (from coralscapesScripts/ project root):

    python scripts/sponges_evaluate.py \
        --preds   work_dir/sponges_pred_20260303_140245 \
        --labels  Sponges/Sponges_18167_sp10/gtFine/test \
        [--output work_dir/sponges_pred_20260303_140245/eval_report.csv] \
        [--ignore-index 255]

Metrics computed per class:
    IoU  (Intersection over Union)  ← primary segmentation metric
    F1   (Dice coefficient)
    Precision
    Recall

Also reports:
    Mean IoU (mIoU) over present classes
    Overall Pixel Accuracy
"""

import argparse
import os
import glob
import csv
import numpy as np
from PIL import Image

# ─── Class definitions ───────────────────────────────────────────────────────
ID2LABEL = {
    0: "unlabeled",
    1: "Ball Yellow Papillate Irregular",
    2: "Cup Beige Thick",
    3: "Cup Black Smooth",
    4: "Cup Orange",
    5: "Cup Red Smooth",
    6: "Cup Red Thick",
    7: "Cup Yellow",
    8: "Fan Pink",
    9: "Massive Purple",
}
N_CLASSES = len(ID2LABEL)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray,
                             n_classes: int, ignore_index: int = 255) -> np.ndarray:
    """Returns n_classes × n_classes confusion matrix."""
    valid = (gt != ignore_index)
    pred  = pred[valid].astype(np.int64)
    gt    = gt[valid].astype(np.int64)

    # Clamp predictions to valid range
    pred = np.clip(pred, 0, n_classes - 1)

    cm = np.bincount(n_classes * gt + pred, minlength=n_classes ** 2)
    return cm.reshape(n_classes, n_classes)


def metrics_from_cm(cm: np.ndarray):
    """Per-class IoU, F1, Precision, Recall from confusion matrix."""
    tp  = np.diag(cm)
    fp  = cm.sum(0) - tp       # predicted as class c, actually something else
    fn  = cm.sum(1) - tp       # actually class c, predicted as something else

    iou       = tp / np.maximum(tp + fp + fn, 1e-9)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall    = tp / np.maximum(tp + fn, 1e-9)
    f1        = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

    pixel_acc = tp.sum() / max(cm.sum(), 1)

    return iou, f1, precision, recall, float(pixel_acc)


def find_label_for_pred(pred_path: str, label_dir: str) -> str | None:
    """
    Given a path like .../PR_..._LC16_pred.png,
    find the matching .../PR_..._LC16_labelIds.png in label_dir.
    """
    base = os.path.basename(pred_path)               # PR_..._LC16_pred.png
    stem = base.replace("_pred.png", "")             # PR_..._LC16
    label_name = f"{stem}_labelIds.png"
    candidate = os.path.join(label_dir, label_name)
    if os.path.isfile(candidate):
        return candidate
    # Recursive search fallback
    matches = glob.glob(os.path.join(label_dir, "**", label_name), recursive=True)
    return matches[0] if matches else None


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Sponges segmentation predictions")
    p.add_argument("--preds",         required=True,  help="Directory containing *_pred.png files")
    p.add_argument("--labels",        required=True,  help="Directory containing *_labelIds.png files")
    p.add_argument("--output",        default=None,   help="Output CSV path (default: <preds_dir>/eval_report.csv)")
    p.add_argument("--ignore-index",  type=int, default=255, help="Label value to ignore (default: 255)")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    pred_dir  = os.path.abspath(args.preds)
    label_dir = os.path.abspath(args.labels)
    ignore    = args.ignore_index

    output_csv = args.output or os.path.join(pred_dir, "eval_report.csv")
    output_txt = output_csv.replace(".csv", "_summary.txt")

    # Find all prediction files
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "**", "*_pred.png"), recursive=True))
    print(f"Found {len(pred_paths)} prediction files")

    # Accumulate confusion matrix across all images
    global_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    per_image_rows = []
    missing = []

    for pred_path in pred_paths:
        label_path = find_label_for_pred(pred_path, label_dir)
        if label_path is None:
            print(f"  [WARN] No ground-truth found for: {os.path.basename(pred_path)}")
            missing.append(pred_path)
            continue

        pred_arr  = load_mask(pred_path)
        label_arr = load_mask(label_path)

        # If sizes differ (e.g. from sliding window padding), resize pred to label size
        if pred_arr.shape != label_arr.shape:
            pred_img  = Image.fromarray(pred_arr.astype(np.uint8))
            pred_arr  = np.array(pred_img.resize(
                (label_arr.shape[1], label_arr.shape[0]),
                resample=Image.NEAREST
            ))

        cm = compute_confusion_matrix(pred_arr, label_arr, N_CLASSES, ignore)
        global_cm += cm

        # Per-image metrics
        iou, f1, prec, rec, px_acc = metrics_from_cm(cm)
        per_image_rows.append({
            "image":        os.path.basename(pred_path).replace("_pred.png", ""),
            "pixel_acc":    float(px_acc),
            "mean_iou":     float(np.nanmean(iou[iou > 0])) if (iou > 0).any() else 0.0,
            "mean_f1":      float(np.nanmean(f1[f1 > 0]))   if (f1 > 0).any()  else 0.0,
            **{f"iou_{ID2LABEL[c]}":  float(iou[c])  for c in range(N_CLASSES)},
            **{f"f1_{ID2LABEL[c]}":   float(f1[c])   for c in range(N_CLASSES)},
            **{f"prec_{ID2LABEL[c]}": float(prec[c]) for c in range(N_CLASSES)},
            **{f"rec_{ID2LABEL[c]}":  float(rec[c])  for c in range(N_CLASSES)},
        })

    if not per_image_rows:
        print("[ERROR] No matched prediction/label pairs found. Check --preds and --labels paths.")
        return

    # ── Global metrics ────────────────────────────────────────────────────────
    g_iou, g_f1, g_prec, g_rec, g_px_acc = metrics_from_cm(global_cm)

    # Only count classes that actually appear in GT
    present = (global_cm.sum(1) > 0)
    miou    = float(np.mean(g_iou[present]))
    mf1     = float(np.mean(g_f1[present]))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = list(per_image_rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"\nPer-image CSV saved to: {output_csv}")

    # ── Print and save summary ────────────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("  SPONGES SEGMENTATION EVALUATION REPORT")
    lines.append("=" * 65)
    lines.append(f"  Predictions : {pred_dir}")
    lines.append(f"  Labels      : {label_dir}")
    lines.append(f"  Images eval : {len(per_image_rows)}  (skipped: {len(missing)})")
    lines.append("")
    lines.append(f"  Overall Pixel Accuracy : {g_px_acc*100:.2f}%")
    lines.append(f"  Mean IoU (mIoU)        : {miou*100:.2f}%")
    lines.append(f"  Mean F1                : {mf1*100:.2f}%")
    lines.append("")
    lines.append("─" * 65)
    lines.append(f"  {'Class':<40} {'IoU':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'Present'}")
    lines.append("─" * 65)
    for c in range(N_CLASSES):
        present_c = global_cm[c].sum() > 0
        lines.append(
            f"  {ID2LABEL[c]:<40} {g_iou[c]*100:>6.2f}%  {g_f1[c]*100:>6.2f}%"
            f"  {g_prec[c]*100:>6.2f}%  {g_rec[c]*100:>6.2f}%  {'✓' if present_c else '✗'}"
        )
    lines.append("─" * 65)
    lines.append("")

    summary = "\n".join(lines)
    print(summary)

    with open(output_txt, "w") as f:
        f.write(summary)
    print(f"Summary saved to: {output_txt}")


if __name__ == "__main__":
    main()
