import os, json
import numpy as np
from PIL import Image, ImageDraw

COCO_JSON = "../Sponges_18167/all.json"
IMG_DIR   = "../Sponges_18167/all"
OUT_MASKS = "../Sponges_18167/gtFine_all"   # output folder for masks

os.makedirs(OUT_MASKS, exist_ok=True)

coco = json.load(open(COCO_JSON))
imgs = {im["id"]: im for im in coco["images"]}

# map your category ids (1..9) -> train ids (0..8)
cat_ids = [c["id"] for c in coco["categories"]]
cat_ids_sorted = sorted(cat_ids)
cat2train = {cid: i for i, cid in enumerate(cat_ids_sorted)}

# group annotations by image
ann_by_img = {}
for ann in coco["annotations"]:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

for image_id, im in imgs.items():
    w, h = im["width"], im["height"]
    mask = np.zeros((h, w), dtype=np.uint8)  # 0..8 classes (background=0 if you want separate bg, see note below)

    # draw each polygon into the mask
    for ann in ann_by_img.get(image_id, []):
        cid = ann["category_id"]
        cls = cat2train[cid]
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):  # polygons
            for poly in seg:
                xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                m = Image.fromarray(mask)
                draw = ImageDraw.Draw(m)
                draw.polygon(xy, fill=int(cls))
                mask = np.array(m, dtype=np.uint8)

    out_name = os.path.splitext(im["file_name"])[0] + "_labelIds.png"
    Image.fromarray(mask).save(os.path.join(OUT_MASKS, out_name))

print("Done. Wrote masks to", OUT_MASKS)
