"""
Stratified re-split of the Sponges_18167 dataset.

Strategy:
  - Analyse every image's annotation to count pixels per sponge class.
  - Use multi-label stratification: sort images so that rare classes appear
    proportionally in all three splits (train / val / test).
  - Writes the new dataset into Sponges_18167_sp10_strat/ (same format as sp10).

Usage (from coralscapesScripts/ project root):
    python scripts/prepare_sponges_stratified.py

Requirements:
    pip install scikit-multilearn   # only needed for skmultilearn, otherwise
    # we use our own greedy stratification which needs only numpy
"""

import os
import json
import shutil
import random
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

# ─── Config ──────────────────────────────────────────────────────────────────
COCO_JSON   = "../Sponges/Sponges_18167/all.json"
IMG_SRC     = "../Sponges/Sponges_18167/all"
OUT_ROOT    = "../Sponges/Sponges_18167_sp10_strat"   # output directory

SEED        = 42
SPLIT       = (0.70, 0.15, 0.15)             # train / val / test

N_SPONGE_CLASSES = 9   # classes 1..9  (0 = background)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def make_dirs(root):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "leftImg8bit", split), exist_ok=True)
        os.makedirs(os.path.join(root, "gtFine",     split), exist_ok=True)


def rasterize_mask(coco, image_id, cat2train, h, w):
    """Return a uint8 H×W mask with background=0 and sponge classes 1..9."""
    ann_by_img = getattr(rasterize_mask, "_cache", None)
    if ann_by_img is None:
        ann_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_img[ann["image_id"]].append(ann)
        rasterize_mask._cache = ann_by_img

    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in ann_by_img.get(image_id, []):
        cls = int(cat2train[ann["category_id"]])
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                m = Image.fromarray(mask)
                draw = ImageDraw.Draw(m)
                draw.polygon(xy, fill=cls)
                mask = np.array(m, dtype=np.uint8)
    return mask


def class_presence_vector(mask, n_classes):
    """Binary vector: which sponge classes (1..n_classes) are present in this mask?"""
    vec = np.zeros(n_classes, dtype=np.int32)
    for c in range(1, n_classes + 1):
        if (mask == c).any():
            vec[c - 1] = 1
    return vec


def greedy_stratified_split(image_ids, presence_matrix, split_ratios, seed):
    """
    Greedy multi-label stratified split.
    For each class, sort images that have it and distribute round-robin
    to splits in proportion to split_ratios.

    Returns dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    rng = random.Random(seed)
    n   = len(image_ids)
    splits = {"train": [], "val": [], "test": []}
    assigned = set()

    n_train = int(split_ratios[0] * n)
    n_val   = int(split_ratios[1] * n)

    # Sort classes from rarest to most common
    class_counts = presence_matrix.sum(axis=0)
    class_order  = np.argsort(class_counts)

    for cls_idx in class_order:
        candidates = [i for i in range(n)
                      if presence_matrix[i, cls_idx] == 1 and image_ids[i] not in assigned]
        rng.shuffle(candidates)

        n_c       = len(candidates)
        n_c_train = max(1, int(split_ratios[0] * n_c))
        n_c_val   = max(1, int(split_ratios[1] * n_c))

        for j, idx in enumerate(candidates):
            if j < n_c_train:
                splits["train"].append(image_ids[idx])
            elif j < n_c_train + n_c_val:
                splits["val"].append(image_ids[idx])
            else:
                splits["test"].append(image_ids[idx])
            assigned.add(image_ids[idx])

    # Assign remaining images (no sponge annotations) to train
    remaining = [image_ids[i] for i in range(n) if image_ids[i] not in assigned]
    rng.shuffle(remaining)
    n_rem_train = max(0, n_train - len(splits["train"]))
    n_rem_val   = max(0, n_val   - len(splits["val"]))
    splits["train"].extend(remaining[:n_rem_train])
    splits["val"].extend(remaining[n_rem_train:n_rem_train + n_rem_val])
    splits["test"].extend(remaining[n_rem_train + n_rem_val:])

    return splits


def print_split_stats(split_ids, presence_matrix, image_ids, id2idx):
    """Print class presence count per split."""
    id2label = {
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
    print(f"\n{'Class':<42} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("─" * 68)
    for c in range(N_SPONGE_CLASSES):
        counts = {}
        for split_name, ids in split_ids.items():
            counts[split_name] = sum(
                1 for img_id in ids if presence_matrix[id2idx[img_id], c] == 1
            )
        total = sum(counts.values())
        label = id2label.get(c + 1, f"class_{c+1}")
        print(f"  {label:<40} {counts['train']:>7} {counts['val']:>7} {counts['test']:>7} {total:>7}")
    print("─" * 68)
    total_imgs = {k: len(v) for k, v in split_ids.items()}
    print(f"  {'TOTAL IMAGES':<40} {total_imgs['train']:>7} {total_imgs['val']:>7} {total_imgs['test']:>7} {sum(total_imgs.values()):>7}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading COCO annotations …")
    with open(COCO_JSON) as f:
        coco = json.load(f)

    imgs      = {im["id"]: im for im in coco["images"]}
    cat_ids   = sorted([c["id"] for c in coco["categories"]])
    cat2train = {cid: i + 1 for i, cid in enumerate(cat_ids)}   # 1..9

    image_ids = list(imgs.keys())
    id2idx    = {img_id: i for i, img_id in enumerate(image_ids)}

    # ── Build presence matrix ─────────────────────────────────────────────────
    print(f"Analysing {len(image_ids)} images for class presence …")
    presence_matrix = np.zeros((len(image_ids), N_SPONGE_CLASSES), dtype=np.int32)

    for i, image_id in enumerate(image_ids):
        im = imgs[image_id]
        mask = rasterize_mask(coco, image_id, cat2train, im["height"], im["width"])
        presence_matrix[i] = class_presence_vector(mask, N_SPONGE_CLASSES)

    # ── Stratified split ─────────────────────────────────────────────────────
    print("Running greedy stratified split …")
    split_ids = greedy_stratified_split(image_ids, presence_matrix, SPLIT, SEED)

    print("\n=== Split Statistics (# images with each class present) ===")
    print_split_stats(split_ids, presence_matrix, image_ids, id2idx)

    # ── Write dataset ────────────────────────────────────────────────────────
    print(f"Writing dataset to: {OUT_ROOT}")
    make_dirs(OUT_ROOT)

    for split_name, ids in split_ids.items():
        print(f"  Writing {split_name} ({len(ids)} images) …")
        for image_id in ids:
            im = imgs[image_id]
            fn = im["file_name"]
            h, w = im["height"], im["width"]

            mask = rasterize_mask(coco, image_id, cat2train, h, w)

            # Copy image
            src_img = os.path.join(IMG_SRC, fn)
            dst_img = os.path.join(OUT_ROOT, "leftImg8bit", split_name, fn)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"    [WARN] Image not found: {src_img}")

            # Save mask
            base      = os.path.splitext(fn)[0]
            mask_name = base + "_labelIds.png"
            dst_mask  = os.path.join(OUT_ROOT, "gtFine", split_name, mask_name)
            Image.fromarray(mask).save(dst_mask)

    # ── Save classes.json and colors.json ────────────────────────────────────
    classes_json = {
        "Ball Yellow Papillate Irregular": 1,
        "Cup Beige Thick": 2,
        "Cup Black Smooth": 3,
        "Cup Orange": 4,
        "Cup Red Smooth": 5,
        "Cup Red Thick": 6,
        "Cup Yellow": 7,
        "Fan Pink": 8,
        "Massive Purple": 9,
    }
    colors_json = {
        "Ball Yellow Papillate Irregular": [255, 255, 0],
        "Cup Beige Thick":                 [245, 245, 220],
        "Cup Black Smooth":                [50,  50,  50],
        "Cup Orange":                      [255, 165, 0],
        "Cup Red Smooth":                  [255, 0,   0],
        "Cup Red Thick":                   [139, 0,   0],
        "Cup Yellow":                      [255, 215, 0],
        "Fan Pink":                        [255, 192, 203],
        "Massive Purple":                  [128, 0,   128],
    }
    with open(os.path.join(OUT_ROOT, "classes.json"), "w") as f:
        json.dump(classes_json, f, indent=4)
    with open(os.path.join(OUT_ROOT, "colors.json"), "w") as f:
        json.dump(colors_json, f, indent=4)

    print("\nDone! Dataset written to:", OUT_ROOT)
    print("Next steps:")
    print("  1. Update configs/segformer-mit-b2-sponges_sp10.yaml → data.root to point here")
    print("  2. Set weight: True in the data section to enable class weights")
    print("  3. Re-run training: python benchmark_runs/train.py --config configs/segformer-mit-b2-sponges_sp10.yaml")


if __name__ == "__main__":
    main()
