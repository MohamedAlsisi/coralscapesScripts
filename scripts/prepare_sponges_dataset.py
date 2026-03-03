import os, json, random, shutil
import numpy as np
from PIL import Image, ImageDraw

COCO_JSON = "../Sponges/Sponges_18167/all.json"
IMG_SRC   = "../Sponges/Sponges_18167/all"

OUT_ROOT_9  = "../Sponges/Sponges_18167_sp9"   # 9-class (ignore unlabeled)
OUT_ROOT_10 = "../Sponges/Sponges_18167_sp10"  # 10-class (background + 9 sponges)

SEED = 42
SPLIT = (0.8, 0.1, 0.1)  # train, val, test

def make_dirs(root):
    for split in ["train","val","test"]:
        os.makedirs(os.path.join(root, "leftImg8bit", split), exist_ok=True)
        os.makedirs(os.path.join(root, "gtFine", split), exist_ok=True)

def rasterize(coco, out_root, mode):
    """
    mode='ignore' -> unlabeled pixels = 255, classes = 0..8
    mode='bg'     -> unlabeled pixels = 0 (background), sponge classes = 1..9
    """
    make_dirs(out_root)

    imgs = {im["id"]: im for im in coco["images"]}
    # category ids -> stable order
    cat_ids = sorted([c["id"] for c in coco["categories"]])

    if mode == "ignore":
        cat2train = {cid: i for i, cid in enumerate(cat_ids)}  # 0..8
    elif mode == "bg":
        cat2train = {cid: i+1 for i, cid in enumerate(cat_ids)}  # 1..9 (0 reserved for bg)
    else:
        raise ValueError("mode must be 'ignore' or 'bg'")

    # group annotations by image_id
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # split images randomly
    ids = list(imgs.keys())
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    n_train = int(SPLIT[0]*n)
    n_val   = int(SPLIT[1]*n)
    split_ids = {
        "train": ids[:n_train],
        "val":   ids[n_train:n_train+n_val],
        "test":  ids[n_train+n_val:],
    }

    for split, image_ids in split_ids.items():
        for image_id in image_ids:
            im = imgs[image_id]
            fn = im["file_name"]
            w, h = im["width"], im["height"]

            # init mask
            if mode == "ignore":
                mask = np.full((h, w), 255, dtype=np.uint8)  # ignore by default
            else:
                mask = np.zeros((h, w), dtype=np.uint8)      # background by default

            # paint polygons
            for ann in ann_by_img.get(image_id, []):
                cid = ann["category_id"]
                cls = int(cat2train[cid])
                seg = ann.get("segmentation", [])
                if isinstance(seg, list):  # polygons
                    for poly in seg:
                        xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                        m = Image.fromarray(mask)
                        draw = ImageDraw.Draw(m)
                        draw.polygon(xy, fill=cls)
                        mask = np.array(m, dtype=np.uint8)

            # copy image
            src_img = os.path.join(IMG_SRC, fn)
            dst_img = os.path.join(out_root, "leftImg8bit", split, fn)
            shutil.copy2(src_img, dst_img)

            # save mask
            base = os.path.splitext(fn)[0]
            mask_name = base + "_labelIds.png"
            dst_mask = os.path.join(out_root, "gtFine", split, mask_name)
            Image.fromarray(mask).save(dst_mask)

    print(f"Done {mode}: wrote dataset to {out_root}")
    print("Counts:", {k: len(v) for k,v in split_ids.items()})

if __name__ == "__main__":
    coco = json.load(open(COCO_JSON))
    rasterize(coco, OUT_ROOT_9, mode="ignore")
    rasterize(coco, OUT_ROOT_10, mode="bg")
