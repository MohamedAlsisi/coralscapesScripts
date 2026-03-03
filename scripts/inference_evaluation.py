import torch 
import albumentations as A
from coralscapesScripts.segmentation.model import Benchmark_Run
from coralscapesScripts.io import setup_config, get_parser, update_config_with_args

from coralscapesScripts.datasets.preprocess import preprocess_inference
from coralscapesScripts.segmentation.model import predict
import glob
import os
import sys
import datetime
import numpy as np
from PIL import Image

# ─── Experiment timing ───────────────────────────────────────────────────────
start_time = datetime.datetime.now()
command_used = " ".join(sys.argv)

# ─── Device setup ────────────────────────────────────────────────────────────
device_count = torch.cuda.device_count()
gpu_info = []
for i in range(device_count):
    name = torch.cuda.get_device_name(i)
    print(f"CUDA Device {i}: {name}")
    gpu_info.append(f"  Device {i}: {name}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ─── Class mappings ──────────────────────────────────────────────────────────
id2label = {"0": "unlabeled", "1": "seagrass", "2": "trash", "3": "other coral dead", "4": "other coral bleached", "5": "sand", "6": "other coral alive", "7": "human", "8": "transect tools", "9": "fish", "10": "algae covered substrate", "11": "other animal", "12": "unknown hard substrate", "13": "background", "14": "dark", "15": "transect line", "16": "massive/meandering bleached", "17": "massive/meandering alive", "18": "rubble", "19": "branching bleached", "20": "branching dead", "21": "millepora", "22": "branching alive", "23": "massive/meandering dead", "24": "clam", "25": "acropora alive", "26": "sea cucumber", "27": "turbinaria", "28": "table acropora alive", "29": "sponge", "30": "anemone", "31": "pocillopora alive", "32": "table acropora dead", "33": "meandering bleached", "34": "stylophora alive", "35": "sea urchin", "36": "meandering alive", "37": "meandering dead", "38": "crown of thorn", "39": "dead clam"}
label2color = {"unlabeled": [255, 255, 255], "human": [255, 0, 0], "background": [29, 162, 216], "fish": [255, 255, 0], "sand": [194, 178, 128], "rubble": [161, 153, 128], "unknown hard substrate": [125, 125, 125], "algae covered substrate": [125, 163, 125], "dark": [31, 31, 31], "branching bleached": [252, 231, 240], "branching dead": [123, 50, 86], "branching alive": [226, 91, 157], "stylophora alive": [255, 111, 194], "pocillopora alive": [255, 146, 150], "acropora alive": [236, 128, 255], "table acropora alive": [189, 119, 255], "table acropora dead": [85, 53, 116], "millepora": [244, 150, 115], "turbinaria": [228, 255, 119], "other coral bleached": [250, 224, 225], "other coral dead": [114, 60, 61], "other coral alive": [224, 118, 119], "massive/meandering alive": [236, 150, 21], "massive/meandering dead": [134, 86, 18], "massive/meandering bleached": [255, 248, 228], "meandering alive": [230, 193, 0], "meandering dead": [119, 100, 14], "meandering bleached": [251, 243, 216], "transect line": [0, 255, 0], "transect tools": [8, 205, 12], "sea urchin": [0, 142, 255], "sea cucumber": [0, 231, 255], "anemone": [0, 255, 189], "sponge": [240, 80, 80], "clam": [189, 255, 234], "other animal": [0, 255, 255], "trash": [255, 0, 134], "seagrass": [125, 222, 125], "crown of thorn": [179, 245, 234], "dead clam": [89, 155, 134]}
id2color = {int(k): label2color[v] for k, v in id2label.items()}

# ─── Config & args ───────────────────────────────────────────────────────────
parser = get_parser()
args = parser.parse_args()

cfg_base_path = '../configs/base.yaml'
cfg = setup_config(args.config, cfg_base_path)
cfg = update_config_with_args(cfg, args)

transform = A.Compose([getattr(A, transform_name)(**transform_params) for transform_name, transform_params
                                                                                in cfg.augmentation["test"].items()])

benchmark_run = Benchmark_Run(run_name = cfg.run_name, model_name = cfg.model.name, 
                                    N_classes = len(id2label), device= device, 
                                    model_kwargs = cfg.model.kwargs,
                                    model_checkpoint = cfg.model.checkpoint,
                                    lora_kwargs = cfg.lora,
                                    training_hyperparameters = cfg.training)
benchmark_run.model.to(device)
benchmark_run.model.eval()

# ─── Output directory: work_dir/<run_name>_<timestamp>/ ──────────────────────
input_dir = args.inputs
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")

if args.outputs:
    output_dir = args.outputs
else:
    work_dir = os.path.join(os.path.dirname(__file__), '..', 'work_dir')
    output_dir = os.path.join(work_dir, f"{cfg.run_name}_{timestamp_str}")

output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)

# ─── Collect images ──────────────────────────────────────────────────────────
if os.path.isfile(input_dir):
    image_paths = [input_dir]
else:
    image_paths = (glob.glob(f'{input_dir}/**/*.png', recursive=True) +
                   glob.glob(f'{input_dir}/**/*.jpg', recursive=True) +
                   glob.glob(f'{input_dir}/**/*.jpeg', recursive=True))

print(f"Found {len(image_paths)} images")
print(f"Outputs will be saved to: {output_dir}")

# ─── Inference loop ──────────────────────────────────────────────────────────
for i, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert('RGB')

    preprocessed_batch, window_dims = preprocess_inference(np.array(image), transform, benchmark_run)
    with torch.no_grad():
        label_pred = predict(preprocessed_batch, 
                                        benchmark_run,
                                        window_dims = window_dims)
        
    label_pred_colors =  np.array([[id2color[pixel] for pixel in row] for row in np.array(label_pred)])
    mask_image = Image.fromarray(label_pred.astype(np.uint8))
    mask_image_colors = Image.fromarray(label_pred_colors.astype(np.uint8), 'RGB')
    overlay = Image.blend(image.convert("RGBA"), mask_image_colors.convert("RGBA"), alpha=0.6)

    # Handle file vs directory for output path mapping
    if os.path.isfile(input_dir):
        out_path = os.path.join(output_dir, os.path.basename(image_path))
    else:
        out_path = image_path.replace(os.path.abspath(input_dir), output_dir)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_image.save(out_path.replace(".png", "_pred.png"))
    overlay.save(out_path.replace(".png", "_overlay.png"))

    print(f"[{i+1}/{len(image_paths)}] {os.path.basename(image_path)}")

# ─── End timing ──────────────────────────────────────────────────────────────
end_time = datetime.datetime.now()
elapsed = end_time - start_time

# ─── Write experiment log ─────────────────────────────────────────────────────
log_path = os.path.join(output_dir, "experiment_log.txt")
with open(log_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("         INFERENCE EVALUATION EXPERIMENT LOG\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Script             : {os.path.abspath(__file__)}\n")
    f.write(f"Command            : python {command_used}\n\n")

    f.write(f"Start time         : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time           : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Elapsed time       : {str(elapsed).split('.')[0]}\n\n")

    f.write("─" * 60 + "\n")
    f.write("MODEL\n")
    f.write("─" * 60 + "\n")
    f.write(f"  Run name         : {cfg.run_name}\n")
    f.write(f"  Model name       : {cfg.model.name}\n")
    f.write(f"  Checkpoint       : {cfg.model.checkpoint}\n")
    f.write(f"  Config file      : {os.path.abspath(args.config)}\n")
    f.write(f"  N classes        : {len(id2label)}\n\n")

    f.write("─" * 60 + "\n")
    f.write("DATA\n")
    f.write("─" * 60 + "\n")
    f.write(f"  Input directory  : {os.path.abspath(input_dir)}\n")
    f.write(f"  Output directory : {output_dir}\n")
    f.write(f"  Images processed : {len(image_paths)}\n")
    f.write(f"  Output files     : {len(image_paths) * 2}  (_pred.png + _overlay.png each)\n\n")

    f.write("─" * 60 + "\n")
    f.write("HARDWARE\n")
    f.write("─" * 60 + "\n")
    f.write(f"  Device           : {device}\n")
    if gpu_info:
        for g in gpu_info:
            f.write(f"{g}\n")
    else:
        f.write("  (CPU only)\n")
    f.write("\n")

    f.write("─" * 60 + "\n")
    f.write("AUGMENTATION (test)\n")
    f.write("─" * 60 + "\n")
    for k, v in cfg.augmentation["test"].items():
        f.write(f"  {k}: {v}\n")
    f.write("\n")

    f.write("─" * 60 + "\n")
    f.write("CLASS MAPPING\n")
    f.write("─" * 60 + "\n")
    for k, v in id2label.items():
        f.write(f"  {k:>3}: {v}\n")
    f.write("\n")

    f.write("=" * 60 + "\n")

print(f"\nExperiment log saved to: {log_path}")