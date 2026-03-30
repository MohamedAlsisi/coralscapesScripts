"""
Evaluate saved coralscapes Lv2 predictions against ground-truth label masks.

Usage (from coralscapesScripts/ project root):

    python scripts/coralscapes_evaluate_lv2.py \\
        --preds   work_dir/coralscapes_lv2_pred_<timestamp> \\
        --labels  coralscapes_lv2/gtFine/test \\
        [--output work_dir/.../eval_report.csv] \\
        [--ignore-index 0]

Outputs:
    eval_report_lv2.csv          – per-image metrics
    eval_report_lv2_summary.txt  – human-readable summary with per-class confusion detail
    confusion_matrix_lv2.csv     – full N×N confusion matrix
"""

import argparse
import os
import glob
import csv
import numpy as np
from PIL import Image

# ─── Lv2 class definitions ───────────────────────────────────────────────────
ID2LABEL = {
    0:  "unlabeled",
    1:  "Algae Covered Substrate",
    2:  "Background",
    3:  "Dark",
    4:  "Fish",
    5:  "Human",
    6:  "Other Animal",
    7:  "Branching Alive",
    8:  "Massive/Meandering Alive",
    9:  "Coral Alive",
    10: "Branching Bleached",
    11: "Massive/Meandering Bleached",
    12: "Coral Bleached",
    13: "Branching Dead",
    14: "Massive/Meandering Dead",
    15: "Coral Dead",
    16: "Sand",
    17: "Seagrass",
    18: "Transect Tools",
    19: "Trash",
    20: "Hard Substrate",
}
N_CLASSES = len(ID2LABEL)  # 21


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray,
                             n_classes: int, ignore_index: int = 0) -> np.ndarray:
    valid = (gt != ignore_index)
    pred  = pred[valid].astype(np.int64)
    gt    = gt[valid].astype(np.int64)
    pred  = np.clip(pred, 0, n_classes - 1)
    cm = np.bincount(n_classes * gt + pred, minlength=n_classes ** 2)
    return cm.reshape(n_classes, n_classes)


def metrics_from_cm(cm: np.ndarray):
    tp  = np.diag(cm)
    fp  = cm.sum(0) - tp
    fn  = cm.sum(1) - tp
    iou       = tp / np.maximum(tp + fp + fn, 1e-9)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall    = tp / np.maximum(tp + fn, 1e-9)
    f1        = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    pixel_acc = tp.sum() / max(cm.sum(), 1)
    return iou, f1, precision, recall, float(pixel_acc)


def top_confusions(cm: np.ndarray, cls_id: int, id2label: dict,
                   ignore_index: int = 0, top_k: int = 3):
    """For a given GT class, return the top-k classes it was predicted AS (excluding correct)."""
    row = cm[cls_id].copy()
    row[cls_id]       = 0
    row[ignore_index] = 0
    total = cm[cls_id].sum()
    if total == 0:
        return []
    results = []
    for idx in np.argsort(row)[::-1][:top_k]:
        count = int(row[idx])
        if count == 0:
            break
        pct = 100.0 * count / max(total, 1)
        results.append((id2label.get(int(idx), f"cls{idx}"), count, pct))
    return results


def find_label_for_pred(pred_path: str, label_dir: str) -> str | None:
    base = os.path.basename(pred_path)
    stem = base.replace("_pred.png", "")
    for label_name in [f"{stem}_gtFine_labelIds.png",
                       f"{stem}_gtFine.png",
                       f"{stem}_labelIds.png"]:
        candidate = os.path.join(label_dir, label_name)
        if os.path.isfile(candidate):
            return candidate
        matches = glob.glob(os.path.join(label_dir, "**", label_name), recursive=True)
        if matches:
            return matches[0]
    return None


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate coralscapes Lv2 segmentation predictions")
    p.add_argument("--preds",        required=True)
    p.add_argument("--labels",       required=True)
    p.add_argument("--output",       default=None)
    p.add_argument("--ignore-index", type=int, default=0)
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    pred_dir  = os.path.abspath(args.preds)
    label_dir = os.path.abspath(args.labels)
    ignore    = args.ignore_index

    output_csv = args.output or os.path.join(pred_dir, "eval_report_lv2.csv")
    output_txt = output_csv.replace(".csv", "_summary.txt")
    output_cm  = os.path.join(os.path.dirname(output_csv), "confusion_matrix_lv2.csv")

    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "**", "*_pred.png"), recursive=True))
    print(f"Found {len(pred_paths)} prediction files")
    if not pred_paths:
        print("[ERROR] No *_pred.png files found in:", pred_dir)
        return

    global_cm      = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    per_image_rows = []
    missing        = []

    for pred_path in pred_paths:
        label_path = find_label_for_pred(pred_path, label_dir)
        if label_path is None:
            print(f"  [WARN] No GT found for: {os.path.basename(pred_path)}")
            missing.append(pred_path)
            continue

        pred_arr  = load_mask(pred_path)
        label_arr = load_mask(label_path)

        if pred_arr.shape != label_arr.shape:
            pred_arr = np.array(
                Image.fromarray(pred_arr.astype(np.uint8)).resize(
                    (label_arr.shape[1], label_arr.shape[0]), resample=Image.NEAREST))

        cm = compute_confusion_matrix(pred_arr, label_arr, N_CLASSES, ignore)
        global_cm += cm

        iou, f1, prec, rec, px_acc = metrics_from_cm(cm)
        per_image_rows.append({
            "image":     os.path.basename(pred_path).replace("_pred.png", ""),
            "pixel_acc": float(px_acc),
            "mean_iou":  float(np.nanmean(iou[iou > 0])) if (iou > 0).any() else 0.0,
            "mean_f1":   float(np.nanmean(f1[f1 > 0]))   if (f1 > 0).any()  else 0.0,
            **{f"iou_{ID2LABEL[c]}":  float(iou[c])  for c in range(N_CLASSES)},
            **{f"f1_{ID2LABEL[c]}":   float(f1[c])   for c in range(N_CLASSES)},
            **{f"prec_{ID2LABEL[c]}": float(prec[c]) for c in range(N_CLASSES)},
            **{f"rec_{ID2LABEL[c]}":  float(rec[c])  for c in range(N_CLASSES)},
        })

    if not per_image_rows:
        print("[ERROR] No matched prediction/label pairs found.")
        return

    g_iou, g_f1, g_prec, g_rec, g_px_acc = metrics_from_cm(global_cm)
    present = (global_cm.sum(1) > 0)
    miou    = float(np.mean(g_iou[present])) if present.any() else 0.0
    mf1     = float(np.mean(g_f1[present]))  if present.any() else 0.0

    # ── Save per-image CSV ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"\nPer-image CSV saved to:       {output_csv}")

    # ── Save confusion matrix CSV ─────────────────────────────────────────────
    labels_list = [ID2LABEL[c] for c in range(N_CLASSES)]
    with open(output_cm, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GT \\ Pred"] + labels_list)
        for r in range(N_CLASSES):
            writer.writerow([labels_list[r]] + [int(global_cm[r, c]) for c in range(N_CLASSES)])
    print(f"Confusion matrix CSV saved to: {output_cm}")

    # ── Build human-readable summary ──────────────────────────────────────────
    W = 74
    lines = []
    lines.append("=" * W)
    lines.append("  CORALSCAPES LV2 — PER-CLASS EVALUATION REPORT")
    lines.append("=" * W)
    lines.append(f"  Predictions  : {pred_dir}")
    lines.append(f"  Labels       : {label_dir}")
    lines.append(f"  Images eval  : {len(per_image_rows)}  (skipped: {len(missing)})")
    lines.append(f"  Ignore index : {ignore}  ({ID2LABEL[ignore]})")
    lines.append("")
    lines.append(f"  Overall Pixel Accuracy : {g_px_acc*100:.2f}%")
    lines.append(f"  Mean IoU (mIoU)        : {miou*100:.2f}%")
    lines.append(f"  Mean F1                : {mf1*100:.2f}%")
    lines.append("")
    lines.append("─" * W)
    lines.append(f"  {'Class':<28} {'GT px':>10}  {'IoU':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    lines.append("─" * W)

    # Sort worst → best by IoU
    class_ids = [c for c in range(N_CLASSES) if c != ignore]
    class_ids.sort(key=lambda c: g_iou[c])

    for c in class_ids:
        gt_px     = int(global_cm[c].sum())
        present_c = gt_px > 0
        flag      = "  ✗ (absent)" if not present_c else ""
        lines.append(
            f"  {ID2LABEL[c]:<28} {gt_px:>10,}  "
            f"{g_iou[c]*100:>5.1f}%  {g_f1[c]*100:>5.1f}%  "
            f"{g_prec[c]*100:>5.1f}%  {g_rec[c]*100:>5.1f}%{flag}"
        )
        for conf_name, conf_cnt, conf_pct in top_confusions(
                global_cm, c, ID2LABEL, ignore_index=ignore, top_k=3):
            lines.append(
                f"    ↳ predicted as  {conf_name:<26} {conf_cnt:>10,} px  ({conf_pct:.1f}% of GT class)")

    lines.append("─" * W)
    lines.append("  Classes sorted by IoU ascending — worst classes appear first.")
    lines.append("")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(output_txt, "w") as f:
        f.write(summary)
    print(f"Summary saved to: {output_txt}")


if __name__ == "__main__":
    main()
