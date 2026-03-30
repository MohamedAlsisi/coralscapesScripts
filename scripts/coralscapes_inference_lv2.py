"""
Inference script for a fine-tuned SegFormer trained on coralscapes_lv2
(20-class Level-2 hierarchy).

Usage (from coralscapesScripts/ project root):
    python scripts/coralscapes_inference_lv2.py \\
        --inputs  coralscapes/leftImg8bit/test \\
        --config  configs/segformer-mit-b2-lv2.yaml \\
        --model-checkpoint <path/to/checkpoint> \\
        [--outputs work_dir/coralscapes_lv2_predictions]

Outputs per image (inside --outputs dir):
    <name>_pred.png      – raw integer label mask (for evaluation)
    <name>_overlay.png   – colour-blended overlay on the original image
    experiment_log.txt   – run metadata
"""

import torch
import albumentations as A
from coralscapesScripts.segmentation.model import Benchmark_Run, predict
from coralscapesScripts.io import setup_config, get_parser, update_config_with_args
from coralscapesScripts.datasets.preprocess import preprocess_inference

import glob
import os
import sys
import datetime
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

start_time   = datetime.datetime.now()
command_used = " ".join(sys.argv)

# ─── Device ──────────────────────────────────────────────────────────────────
device_count = torch.cuda.device_count()
gpu_info = []
for i in range(device_count):
    name = torch.cuda.get_device_name(i)
    print(f"CUDA Device {i}: {name}")
    gpu_info.append(f"  Device {i}: {name}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ─── Lv2 class / colour mapping ──────────────────────────────────────────────
id2label = {
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

id2color = {
    0:  [255, 255, 255],   # unlabeled
    1:  [125, 163, 125],   # Algae Covered Substrate
    2:  [29,  162, 216],   # Background
    3:  [31,  31,  31],    # Dark
    4:  [255, 255, 0],     # Fish
    5:  [255, 0,   0],     # Human
    6:  [0,   255, 255],   # Other Animal
    7:  [226, 91,  157],   # Branching Alive
    8:  [236, 150, 21],    # Massive/Meandering Alive
    9:  [224, 118, 119],   # Coral Alive
    10: [252, 231, 240],   # Branching Bleached
    11: [255, 248, 228],   # Massive/Meandering Bleached
    12: [250, 224, 225],   # Coral Bleached
    13: [123, 50,  86],    # Branching Dead
    14: [134, 86,  18],    # Massive/Meandering Dead
    15: [114, 60,  61],    # Coral Dead
    16: [194, 178, 128],   # Sand
    17: [125, 222, 125],   # Seagrass
    18: [8,   205, 12],    # Transect Tools
    19: [255, 0,   134],   # Trash
    20: [125, 125, 125],   # Hard Substrate
}

N_CLASSES = len(id2label)  # 21 (0..20)

# ─── Config & args ───────────────────────────────────────────────────────────
parser = get_parser()
args   = parser.parse_args()

cfg_base_path = 'configs/base.yaml'
cfg = setup_config(args.config, cfg_base_path)
cfg = update_config_with_args(cfg, args)

if getattr(args, 'model_checkpoint', None):
    cfg.model.checkpoint = args.model_checkpoint

transform = A.Compose([
    getattr(A, k)(**v) for k, v in cfg.augmentation["test"].items()
])

# ─── Load model ──────────────────────────────────────────────────────────────
benchmark_run = Benchmark_Run(
    run_name=cfg.run_name,
    model_name=cfg.model.name,
    N_classes=N_CLASSES,
    device=device,
    model_kwargs=cfg.model.kwargs,
    model_checkpoint=cfg.model.checkpoint,
    lora_kwargs=cfg.lora,
    training_hyperparameters=cfg.training,
)
benchmark_run.model.to(device)
benchmark_run.model.eval()

# ─── Output directory ────────────────────────────────────────────────────────
input_dir     = args.inputs
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")

if args.outputs:
    output_dir = args.outputs
else:
    work_dir   = os.path.join(os.path.dirname(__file__), '..', 'work_dir')
    output_dir = os.path.join(work_dir, f"coralscapes_lv2_pred_{timestamp_str}")

output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)

# ─── Collect images ──────────────────────────────────────────────────────────
if os.path.isfile(input_dir):
    image_paths = [input_dir]
else:
    image_paths = (
        glob.glob(f'{input_dir}/**/*.png',  recursive=True) +
        glob.glob(f'{input_dir}/**/*.jpg',  recursive=True) +
        glob.glob(f'{input_dir}/**/*.jpeg', recursive=True)
    )

if not image_paths:
    print(f"[ERROR] No images found in: {input_dir}")
    sys.exit(1)

print(f"Found {len(image_paths)} images")
print(f"Outputs will be saved to: {output_dir}")


def label_to_color(label_arr):
    h, w = label_arr.shape
    color_arr = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in id2color.items():
        color_arr[label_arr == class_id] = color
    return color_arr


# ─── Inference loop ──────────────────────────────────────────────────────────
for i, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert('RGB')

    preprocessed_batch, window_dims = preprocess_inference(
        np.array(image), transform, benchmark_run
    )

    with torch.no_grad():
        label_pred = predict(preprocessed_batch, benchmark_run, window_dims=window_dims)

    label_pred_np = np.clip(np.array(label_pred), 0, N_CLASSES - 1)
    color_arr     = label_to_color(label_pred_np)

    mask_image        = Image.fromarray(label_pred_np.astype(np.uint8))
    mask_image_colors = Image.fromarray(color_arr, 'RGB')
    overlay           = Image.blend(
        image.resize(mask_image_colors.size).convert("RGBA"),
        mask_image_colors.convert("RGBA"),
        alpha=0.5,
    )

    if os.path.isfile(input_dir):
        out_base = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0])
    else:
        rel      = os.path.relpath(image_path, os.path.abspath(input_dir))
        out_base = os.path.join(output_dir, os.path.splitext(rel)[0])

    os.makedirs(os.path.dirname(out_base) or output_dir, exist_ok=True)
    mask_image.save(f"{out_base}_pred.png")
    overlay.save(f"{out_base}_overlay.png")

    print(f"[{i+1}/{len(image_paths)}] {os.path.basename(image_path)} → saved")

# ─── Experiment log ──────────────────────────────────────────────────────────
end_time = datetime.datetime.now()
elapsed  = end_time - start_time

log_path = os.path.join(output_dir, "experiment_log.txt")
with open(log_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("    CORALSCAPES LV2 INFERENCE EXPERIMENT LOG\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Command            : python {command_used}\n")
    f.write(f"Start time         : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time           : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Elapsed time       : {str(elapsed).split('.')[0]}\n\n")
    f.write("─" * 60 + "\n")
    f.write("MODEL\n")
    f.write("─" * 60 + "\n")
    f.write(f"  Run name         : {cfg.run_name}\n")
    f.write(f"  Model            : {cfg.model.name}\n")
    f.write(f"  Checkpoint       : {cfg.model.checkpoint}\n")
    f.write(f"  N classes        : {N_CLASSES}\n\n")
    f.write("─" * 60 + "\n")
    f.write("DATA\n")
    f.write("─" * 60 + "\n")
    f.write(f"  Input            : {os.path.abspath(input_dir)}\n")
    f.write(f"  Output           : {output_dir}\n")
    f.write(f"  Images processed : {len(image_paths)}\n\n")
    f.write("─" * 60 + "\n")
    f.write("CLASS MAPPING (LV2)\n")
    f.write("─" * 60 + "\n")
    for k, v in id2label.items():
        f.write(f"  {k:>3}: {v:<40} {str(id2color[k])}\n")
    f.write("\n" + "=" * 60 + "\n")

print(f"\nExperiment log saved to: {log_path}")
