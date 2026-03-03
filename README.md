# Coralscapes Scripts

This repository contains the scripts for preparing the Sponges dataset, and training, evaluating, and running inference with Segformer.

## Prerequisites

Ensure you have your environment activated and dependencies installed. All commands should be run from the root of the repository (`/home/mohamed/student/mo/coralscapesScripts`):

```bash
cd /home/mohamed/student/mo/coralscapesScripts
```

## 1. Dataset Preparation

If you need to prepare the stratified dataset from the raw COCO bounding boxes:

```bash
python scripts/prepare_sponges_stratified.py
```

This reads from `Sponges/Sponges_18167` and outputs the stratified dataset to `Sponges/Sponges_18167_sp10_strat`.

## 2. Training

To train the Segformer model using the `sp10_strat` configuration (which uses class weighting and a batch size configured for 24GB GPUs):

```bash
python -u benchmark_runs/train.py \
  --config configs/segformer-mit-b2-sponges_sp10_strat.yaml \
  2>&1 | tee work_dir/sponges_train_strat.log
```

## 3. Inference

To run inference on the test set using your trained weights and output the prediction image masks:

```bash
python scripts/sponges_inference.py \
  --inputs Sponges/Sponges_18167_sp10_strat/leftImg8bit/test/ \
  --config configs/segformer-mit-b2-sponges_sp10_strat.yaml \
  --outputs work_dir/sponges_predictions_strat
```

*Note: This script will output the raw mask arrays, color overlays, and an experiment log into the `--outputs` folder.*

## 4. Evaluation

To evaluate the predictions generated during the inference step against the ground truth labels and compute metrics like mIoU and F1-Scores:

```bash
cd scripts/
python sponges_evaluate.py \
    --preds   ../work_dir/sponges_predictions_strat \
    --labels  ../Sponges/Sponges_18167_sp10_strat/gtFine/test
```
*(The evaluation script is currently designed to be run from inside the `scripts/` directory).*
