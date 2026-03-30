"""
Remap Coralscapes Lv3 (39-class) ground-truth masks to Lv1 (14 classes)
and Lv2 (20 classes) following the class hierarchy from the paper figure.

Reads : coralscapes/gtFine/{train,val,test}/*_gtFine_labelIds.png
Writes: coralscapes_lv1/gtFine/{train,val,test}/*_gtFine_labelIds.png
        coralscapes_lv2/gtFine/{train,val,test}/*_gtFine_labelIds.png
        (images are symlinked, not copied)

Usage (from coralscapesScripts/ project root):
    python scripts/prepare_coralscapes_hierarchical.py [--dry-run]
"""

import os
import json
import glob
import shutil
import argparse
import numpy as np
from PIL import Image

# ─── Lv3 class id (from coralscapes/classes.json) ──────────────────────────
LV3_ID = {
    "seagrass": 1,
    "trash": 2,
    "other coral dead": 3,
    "other coral bleached": 4,
    "sand": 5,
    "other coral alive": 6,
    "human": 7,
    "transect tools": 8,
    "fish": 9,
    "algae covered substrate": 10,
    "other animal": 11,
    "unknown hard substrate": 12,
    "background": 13,
    "dark": 14,
    "transect line": 15,
    "massive/meandering bleached": 16,
    "massive/meandering alive": 17,
    "rubble": 18,
    "branching bleached": 19,
    "branching dead": 20,
    "branching alive": 22,
    "massive/meandering dead": 23,
    "clam": 24,
    "acropora alive": 25,
    "sea cucumber": 26,
    "turbinaria": 27,
    "table acropora alive": 28,
    "sponge": 29,
    "anemone": 30,
    "pocillopora alive": 31,
    "table acropora dead": 32,
    "meandering bleached": 33,
    "stylophora alive": 34,
    "sea urchin": 35,
    "meandering alive": 36,
    "meandering dead": 37,
    "crown of thorn": 38,
    "dead clam": 39,
    "millepora": 21,
}

# ─── Level 1: 14 classes ───────────────────────────────────────────────────
#  0 = unlabeled (pixel value 0 in Lv3 or anything unmapped)
LV1_ID2LABEL = {
    0:  "unlabeled",
    1:  "Algae Covered Substrate",
    2:  "Background",
    3:  "Dark",
    4:  "Fish",
    5:  "Human",
    6:  "Other Animal",
    7:  "Coral Alive",
    8:  "Coral Bleached",
    9:  "Coral Dead",
    10: "Sand",
    11: "Seagrass",
    12: "Transect Tools",
    13: "Trash",
    14: "Hard Substrate",
}

LV1_COLORS = {
    "unlabeled":               [255, 255, 255],
    "Algae Covered Substrate": [125, 163, 125],
    "Background":              [29,  162, 216],
    "Dark":                    [31,  31,  31],
    "Fish":                    [255, 255, 0],
    "Human":                   [255, 0,   0],
    "Other Animal":            [0,   255, 255],
    "Coral Alive":             [226, 91,  157],
    "Coral Bleached":          [252, 231, 240],
    "Coral Dead":              [114, 60,  61],
    "Sand":                    [194, 178, 128],
    "Seagrass":                [125, 222, 125],
    "Transect Tools":          [8,   205, 12],
    "Trash":                   [255, 0,   134],
    "Hard Substrate":          [125, 125, 125],
}

# Lv3 id → Lv1 id
LV3_TO_LV1 = {
    LV3_ID["algae covered substrate"]: 1,
    LV3_ID["background"]:              2,
    LV3_ID["dark"]:                    3,
    LV3_ID["fish"]:                    4,
    LV3_ID["human"]:                   5,
    # Other Animal group
    LV3_ID["other animal"]:            6,
    LV3_ID["sea urchin"]:              6,
    LV3_ID["sea cucumber"]:            6,
    LV3_ID["anemone"]:                 6,
    LV3_ID["sponge"]:                  6,
    LV3_ID["clam"]:                    6,
    LV3_ID["crown of thorn"]:          6,
    LV3_ID["dead clam"]:               6,
    # Coral Alive group
    LV3_ID["branching alive"]:         7,
    LV3_ID["stylophora alive"]:        7,
    LV3_ID["pocillopora alive"]:       7,
    LV3_ID["acropora alive"]:          7,
    LV3_ID["table acropora alive"]:    7,
    LV3_ID["millepora"]:               7,
    LV3_ID["massive/meandering alive"]:7,
    LV3_ID["meandering alive"]:        7,
    LV3_ID["turbinaria"]:              7,
    LV3_ID["other coral alive"]:       7,
    # Coral Bleached group
    LV3_ID["branching bleached"]:      8,
    LV3_ID["massive/meandering bleached"]: 8,
    LV3_ID["meandering bleached"]:     8,
    LV3_ID["other coral bleached"]:    8,
    # Coral Dead group
    LV3_ID["branching dead"]:          9,
    LV3_ID["table acropora dead"]:     9,
    LV3_ID["massive/meandering dead"]: 9,
    LV3_ID["meandering dead"]:         9,
    LV3_ID["other coral dead"]:        9,
    # Substrate
    LV3_ID["sand"]:                    10,
    LV3_ID["seagrass"]:                11,
    LV3_ID["transect line"]:           12,
    LV3_ID["transect tools"]:          12,
    LV3_ID["trash"]:                   13,
    LV3_ID["rubble"]:                  14,
    LV3_ID["unknown hard substrate"]:  14,
}

# ─── Level 2: 20 classes ────────────────────────────────────────────────────
LV2_ID2LABEL = {
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

LV2_COLORS = {
    "unlabeled":                    [255, 255, 255],
    "Algae Covered Substrate":      [125, 163, 125],
    "Background":                   [29,  162, 216],
    "Dark":                         [31,  31,  31],
    "Fish":                         [255, 255, 0],
    "Human":                        [255, 0,   0],
    "Other Animal":                 [0,   255, 255],
    "Branching Alive":              [226, 91,  157],
    "Massive/Meandering Alive":     [236, 150, 21],
    "Coral Alive":                  [224, 118, 119],
    "Branching Bleached":           [252, 231, 240],
    "Massive/Meandering Bleached":  [255, 248, 228],
    "Coral Bleached":               [250, 224, 225],
    "Branching Dead":               [123, 50,  86],
    "Massive/Meandering Dead":      [134, 86,  18],
    "Coral Dead":                   [114, 60,  61],
    "Sand":                         [194, 178, 128],
    "Seagrass":                     [125, 222, 125],
    "Transect Tools":               [8,   205, 12],
    "Trash":                        [255, 0,   134],
    "Hard Substrate":               [125, 125, 125],
}

# Lv3 id → Lv2 id
LV3_TO_LV2 = {
    LV3_ID["algae covered substrate"]: 1,
    LV3_ID["background"]:              2,
    LV3_ID["dark"]:                    3,
    LV3_ID["fish"]:                    4,
    LV3_ID["human"]:                   5,
    # Other Animal
    LV3_ID["other animal"]:            6,
    LV3_ID["sea urchin"]:              6,
    LV3_ID["sea cucumber"]:            6,
    LV3_ID["anemone"]:                 6,
    LV3_ID["sponge"]:                  6,
    LV3_ID["clam"]:                    6,
    LV3_ID["crown of thorn"]:          6,
    LV3_ID["dead clam"]:               6,
    # Branching Alive
    LV3_ID["branching alive"]:         7,
    LV3_ID["stylophora alive"]:        7,
    LV3_ID["pocillopora alive"]:       7,
    LV3_ID["acropora alive"]:          7,
    LV3_ID["table acropora alive"]:    7,
    LV3_ID["millepora"]:               7,
    # Massive/Meandering Alive
    LV3_ID["massive/meandering alive"]:8,
    LV3_ID["meandering alive"]:        8,
    LV3_ID["turbinaria"]:              8,
    # Coral Alive (other)
    LV3_ID["other coral alive"]:       9,
    # Branching Bleached
    LV3_ID["branching bleached"]:      10,
    # Massive/Meandering Bleached
    LV3_ID["massive/meandering bleached"]: 11,
    LV3_ID["meandering bleached"]:     11,
    # Coral Bleached (other)
    LV3_ID["other coral bleached"]:    12,
    # Branching Dead
    LV3_ID["branching dead"]:          13,
    LV3_ID["table acropora dead"]:     13,
    # Massive/Meandering Dead
    LV3_ID["massive/meandering dead"]: 14,
    LV3_ID["meandering dead"]:         14,
    # Coral Dead (other)
    LV3_ID["other coral dead"]:        15,
    # Substrate
    LV3_ID["sand"]:                    16,
    LV3_ID["seagrass"]:                17,
    LV3_ID["transect line"]:           18,
    LV3_ID["transect tools"]:          18,
    LV3_ID["trash"]:                   19,
    LV3_ID["rubble"]:                  20,
    LV3_ID["unknown hard substrate"]:  20,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
SRC_ROOT  = "coralscapes"
DST_ROOTS = {
    "lv1": ("coralscapes_lv1", LV3_TO_LV1, LV1_ID2LABEL, LV1_COLORS),
    "lv2": ("coralscapes_lv2", LV3_TO_LV2, LV2_ID2LABEL, LV2_COLORS),
}
SPLITS = ["train", "val", "test"]


def build_lut(remap: dict, max_id: int = 256) -> np.ndarray:
    """Build a lookup table (array) for fast pixel remapping via numpy indexing."""
    lut = np.zeros(max_id, dtype=np.uint8)
    for src_id, dst_id in remap.items():
        if src_id < max_id:
            lut[src_id] = dst_id
    return lut


def remap_mask(mask_arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply lookup table to a uint8 mask array."""
    clipped = np.clip(mask_arr, 0, len(lut) - 1).astype(np.uint8)
    return lut[clipped]


def write_metadata(dst_root: str, id2label: dict, colors: dict):
    """Write classes.json and colors.json."""
    # classes.json: {name: id} for non-unlabeled classes
    classes_out = {v: k for k, v in id2label.items() if k != 0}
    colors_out  = {name: colors[name] for name in colors if name != "unlabeled"}
    with open(os.path.join(dst_root, "classes.json"), "w") as f:
        json.dump(classes_out, f, indent=4)
    with open(os.path.join(dst_root, "colors.json"), "w") as f:
        json.dump(colors_out, f, indent=4)


def make_dirs(dst_root: str):
    for split in SPLITS:
        os.makedirs(os.path.join(dst_root, "gtFine",     split), exist_ok=True)
        os.makedirs(os.path.join(dst_root, "leftImg8bit", split), exist_ok=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Remap coralscapes Lv3 masks to Lv1 and Lv2.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing any files.")
    args = parser.parse_args()

    for level, (dst_root, remap, id2label, colors) in DST_ROOTS.items():
        print(f"\n{'='*60}")
        print(f"  Processing {level.upper()}: {SRC_ROOT} → {dst_root}")
        print(f"{'='*60}")

        lut = build_lut(remap)
        n_classes_out = max(id2label.keys())

        if not args.dry_run:
            make_dirs(dst_root)

        total_pixels = {k: 0 for k in id2label}
        total_files  = 0

        for split in SPLITS:
            gt_dir  = os.path.join(SRC_ROOT, "gtFine",      split)
            img_dir = os.path.join(SRC_ROOT, "leftImg8bit",  split)

            if not os.path.isdir(gt_dir):
                print(f"  [{split}] Directory not found: {gt_dir}")
                continue

            # Coralscapes uses nested site dirs: gtFine/train/site10/site10_*_gtFine.png
            # Collect all .png files recursively, excluding leftImg8bit images
            gt_masks = glob.glob(os.path.join(gt_dir, "**", "*_gtFine.png"), recursive=True)
            # Fallback: _gtFine_labelIds.png or flat _labelIds.png
            if not gt_masks:
                gt_masks = glob.glob(os.path.join(gt_dir, "**", "*_gtFine_labelIds.png"), recursive=True)
            if not gt_masks:
                gt_masks = glob.glob(os.path.join(gt_dir, "**", "*_labelIds.png"), recursive=True)

            print(f"  [{split}] Found {len(gt_masks)} GT masks")

            for mask_path in sorted(gt_masks):
                mask_arr = np.array(Image.open(mask_path))
                remapped = remap_mask(mask_arr, lut)

                # Stats
                for cls_id, cls_name in id2label.items():
                    total_pixels[cls_id] += int((remapped == cls_id).sum())

                if args.dry_run:
                    continue

                # Save remapped mask (mirror the same relative path under dst_root/gtFine/)
                rel      = os.path.relpath(mask_path, os.path.join(SRC_ROOT, "gtFine"))
                dst_mask = os.path.join(dst_root, "gtFine", rel)
                os.makedirs(os.path.dirname(dst_mask), exist_ok=True)
                Image.fromarray(remapped).save(dst_mask)

                # Find the corresponding leftImg8bit image.
                # Coralscapes: site10_000001_012400_gtFine.png → site10_000001_012400_leftImg8bit.png
                img_candidate = mask_path.replace(
                    os.path.join(SRC_ROOT, "gtFine"),
                    os.path.join(SRC_ROOT, "leftImg8bit")
                ).replace("_gtFine.png", "_leftImg8bit.png") \
                 .replace("_gtFine_labelIds.png", "_leftImg8bit.png") \
                 .replace("_labelIds.png", "_leftImg8bit.png")

                # Fallback: try jpg/jpeg if .png not found
                if not os.path.exists(img_candidate):
                    for ext in [".jpg", ".jpeg"]:
                        alt = img_candidate.replace("_leftImg8bit.png", f"_leftImg8bit{ext}")
                        if os.path.exists(alt):
                            img_candidate = alt
                            break

                if os.path.exists(img_candidate):
                    rel_img = os.path.relpath(img_candidate,
                                              os.path.join(SRC_ROOT, "leftImg8bit"))
                    dst_img = os.path.join(dst_root, "leftImg8bit", rel_img)
                    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                    if not os.path.exists(dst_img):
                        try:
                            os.symlink(os.path.abspath(img_candidate), dst_img)
                        except OSError:
                            shutil.copy2(img_candidate, dst_img)
                else:
                    print(f"    [WARN] Image not found for: {os.path.basename(mask_path)}")

                total_files += 1

        # Print pixel distribution
        total_px = sum(total_pixels.values()) or 1
        print(f"\n  Class pixel distribution ({level.upper()}):")
        print(f"  {'ID':>4}  {'Name':<35} {'Pixels':>12}  {'%':>6}")
        print("  " + "─" * 65)
        for cls_id, cls_name in id2label.items():
            px  = total_pixels[cls_id]
            pct = 100.0 * px / total_px
            print(f"  {cls_id:>4}  {cls_name:<35} {px:>12,}  {pct:>6.2f}%")

        if not args.dry_run:
            write_metadata(dst_root, id2label, colors)
            print(f"\n  ✓ Wrote {total_files} remapped masks to {dst_root}/")
            print(f"  ✓ Wrote classes.json and colors.json")
        else:
            print(f"\n  [DRY-RUN] No files written. Total masks scanned: {total_files}")

    print("\nDone.")


if __name__ == "__main__":
    main()
