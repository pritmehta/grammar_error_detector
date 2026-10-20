# Grammar Error Correction using Transformers

This project implements an **end-to-end Grammar Error Correction (GEC)** system using a **Transformer-based sequence-to-sequence model (T5)**.  
The model learns to correct grammatical mistakes in English sentences and outputs both the corrected sentence and a confidence score.

---

## 🔍 Overview

- **Task:** Grammar Error Correction (wrong → corrected sentence)
- **Model:** T5-small (Transformer, seq2seq)
- **Dataset:** C4-200M (synthetic GEC dataset, 200k sampled sentences)
- **Frameworks:** PyTorch, Hugging Face Transformers
- **Training:** GPU-accelerated (RTX 3050)
- **Output:** Corrected sentence + confidence score

---

##  Project Structure

```
grammar_error_detector/
│
├── src/
│ ├── preprocess.py # Dataset sampling & preprocessing
│ ├── dataset.py # Custom PyTorch dataset
│ ├── model.py # Model & tokenizer loading
│ ├── train.py # Training pipeline (GPU-enabled)
│ └── predict.py # Inference + confidence score
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Setup

### 1️Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Prepration

This project uses a subset (200k samples) from the C4-200M dataset.

## Confidence Score

The confidence score is computed by averaging the token-level probabilities produced by the decoder during generation.
This is a standard approach for estimating confidence in sequence-to-sequence NLP models

---

## Key Features

Transformer-based grammar correction

Large-scale dataset handling via sampling

GPU-accelerated training

Confidence-aware predictions

Clean, modular codebase

Reproducible ML pipeline

---
