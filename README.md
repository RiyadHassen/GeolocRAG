# GeoLocRAG
CS 762 Deep Learning project. Learn to geolocate with retrieval and reasoning.

## Project Summary

GeolocRAG combines retrieval-augmented pipelines and language-model reasoning to infer geographic location from visual and textual clues. The repository contains code for building a retrieval index, finetuning and running reasoning models, inference scripts for both baseline and finetuned models, and evaluation utilities.

## Key Files and Directories

- `edit.py`: Utility for preparing or editing datasets/configurations.
- `eval.py`: Evaluation script — compares predictions to ground truth and computes project metrics (distance error, accuracy@k, etc.).
- `finetune.py`: General finetuning script for model training.
- `finetune_qwen.py`: Qwen-specific finetuning utilities and entrypoints.
- `index.index`: Prebuilt retrieval index — skip rebuilding if up-to-date.
- `infer.py`: Baseline inference script for running model predictions.
- `infer_finetuned.py`: Inference wrapper that loads a finetuned model and runs predictions on inputs.
- `rag_retrive_builder.py`: Build or update the retrieval index from datasets.
- `ragpipe.py`: End-to-end pipeline tying retrieval, reasoning, and inference together.
- `reasoningLM.py`: Wrapper/utilities for the reasoning language model component.
- `train_geo_reasoning.sh`: Shell script to launch geolocation reasoning model training (often contains training flags for distributed runs).
- `qwenvldownload.py`: Helpers to download Qwen-related datasets or checkpoints.
- `finetune_scripts/`: Collection of experimental finetuning notebooks and helper scripts (LoRA, CLIP, etc.).
- `output/`: Contains tokenizer, adapter, and checkpoint artifacts (e.g., `adapter_model.safetensors`, `tokenizer.json`, `checkpoint-*`). These can be used for inference.

## Run / Evaluate — High-level Template

Replace placeholders and confirm each script's CLI flags (they may differ slightly). Example workflow:

1) Prepare environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Or install core packages manually if requirements is missing:
pip install transformers accelerate sentence-transformers datasets peft deepspeed pillow huggingface_hub scipy
```

2) Prepare retrieval index

- If `index.index` exists and is current, you can skip rebuild.
- To (re)build (adjust flags as needed):

```bash
python rag_retrive_builder.py \
	--data-dir /path/to/dataset \
	--out index.index \
	--embedding-model sentence-transformers/your-model
```

3) Run inference (single example or batch)

- Baseline / non-finetuned:

```bash
python infer.py \
	--input path/to/input.jsonl \
	--index path/to/index.index \
	--output preds_baseline.jsonl
```

- Using a finetuned checkpoint:

```bash
python infer_finetuned.py \
	--checkpoint output/checkpoint-4914 \
	--index path/to/index.index \
	--input path/to/input.jsonl \
	--output preds_finetuned.jsonl
```

- Run the full pipeline (if available):

```bash
python ragpipe.py --config configs/my_run.yaml --output pipeline_preds.jsonl
```

4) Evaluate predictions

```bash
python eval.py \
	--pred pipeline_preds.jsonl \
	--gt path/to/ground_truth.jsonl \
	--metrics-out results_summary.json
```

Adjust the CLI flags if `eval.py` uses different argument names — open the script to confirm exact usage.

5) Train / Finetune

- Quick shell-run:

```bash
bash train_geo_reasoning.sh
```

- Python entry (example):

```bash
python finetune.py \
	--train-data path/to/train.jsonl \
	--val-data path/to/val.jsonl \
	--output-dir output/my_finetune \
	--epochs 3 \
	--batch-size 8
```

6) Reproducibility checklist

- Fix random seeds in scripts or config.
- Record which `index.index` and `checkpoint-*` were used (checksums or paths).
- Save `preds_*.jsonl` and `results_summary.json` with experiment metadata.

## Notebooks — 

These are interactive/experimental notebooks that are useful for exploration but are not required for batch or CI-style runs:

- `Navig.ipynb` — exploratory analysis and visualization.
- `test_fintune_infernce.ipynb` — interactive debugging of finetuned inference.
- `test_retrival.ipynb` — retrieval experiments and ad-hoc checks.
- `finetune_scripts/Qwen_finetwen_with_naviclues_datasest.ipynb` — Main experimental run on GoogleColab  LoRA/CLIP finetuning notebooks;

## RAG and Adapter weights

- Confirm exact CLI flags by opening `infer.py`, `infer_finetuned.py`, and `eval.py` before running — script signatures may vary.
- If `index.index` rag vector dataset embeding `Plon kt` Dataset.
- `output/` contains artifacts that can be used directly for inference (tokenizer, adapters, checkpoints).

## Eval

## Quick Sanity-Check Commands

```bash
nvidia-smi
python -c "import transformers,torch; print(transformers.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

python infer.py --input examples/one_example.jsonl --index index.index --output test_out.jsonl
```

## Image Results

This project produces visual outputs to help inspect model predictions. Below are results from finetuned models on the NaviClues dataset:

### Finetuned Qwen2-VLM with NaviClues

![Finetuned Qwen2-VLM with NaviClues](finetuned_qwen2vlm_with_NaciClues.png)

This visualization shows the performance of the finetuned Qwen2-VLM model trained on the NaviClues dataset. The model learns to predict geographic locations from visual clues in the images.

### Madison Counts

![Madison Counts](madison-coconuts.png)

This result displays prediction accuracy and error distribution for the Madison area test set, showing how well the model generalizes to specific geographic regions.

### Output Schema

Typical inference predictions are stored in JSONL format with the following fields:

```json
{
  "id": "example_0001",
  "image_path": "path/to/image.jpg",
  "pred_lat": 43.0757,
  "pred_lon": -89.4070,
  "gt_lat": 43.0731,
  "gt_lon": -89.4012,
  "score": 0.87
}
```

### Visualization Commands

To generate similar visualizations after running inference:

```bash
mkdir -p output/image_results
python tools/visualize_preds.py --input preds_finetuned.jsonl --output output/image_results
```





