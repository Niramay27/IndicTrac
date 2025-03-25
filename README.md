# IndicTrac

# SeamlessM4T and Whisper Fine-Tuning

This repository contains code and scripts for fine-tuning SeamlessM4T and Whisper models, along with dataset preparation and logging.

## Files and Directories

- **`results`**: Logging file containing the SeamlessM4T model logging data.
- **`environment.yaml`**: The current Conda environment requirements.
- **`sM4T.ipynb`**: Jupyter notebook for fine-tuning the SeamlessM4T model.
- **`VAD.ipynb`**: Pipeline for creating the dataset and saving it as a JSONL file.
- **`prepare_custom_dataset.py`**: Script to update the JSONL file as required by the SeamlessM4T model.
- **`whisper.ipynb`**: Jupyter notebook for fine-tuning the Whisper model.

## Setup

To replicate the environment, run:

```bash
conda env create -f environment.yaml
conda activate <env_name>

