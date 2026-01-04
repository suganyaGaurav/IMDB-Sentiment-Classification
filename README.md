# DistilBERT IMDB Sentiment Classifier

## Overview

This repository hosts a **fine-tuned DistilBERT model** for **binary sentiment classification** on the IMDB movie reviews dataset.  
The model predicts whether a given movie review expresses **positive** or **negative** sentiment.

The project is designed as a **lightweight, reproducible NLP pipeline** intended for:
- rapid experimentation
- demonstrations
- baseline sentiment analysis
- learning and evaluation workflows

The emphasis is on **correctness, evaluation, and traceability**, rather than aggressive optimization.

---

## Base Model

- **Model**: `distilbert-base-uncased`
- **Framework**: Hugging Face Transformers
- **Task**: Binary Text Classification (Sentiment Analysis)

DistilBERT was selected to balance **performance, speed, and resource efficiency**, making it suitable for small-scale and CPU/GPU-limited environments.

---

## Training Details

- **Dataset**: IMDB Movie Reviews  
  - Preprocessed externally (HTML removal, normalization)
  - Balanced binary labels (positive / negative)
  - Train / test split applied before fine-tuning

- **Objective**: Binary sentiment classification

- **Optimization**:
  - Adam-based optimizer (via Hugging Face Trainer)
  - Fixed learning rate (`2e-5`)
  - Limited-epoch training for fast iteration (smoke-test setup)

- **Regularization & Stability**:
  - Dropout (as defined in DistilBERT architecture)
  - Gradient clipping (Trainer default behavior)
  - Deterministic seed setup for reproducibility

> Note: Training was intentionally performed on a **subset of the dataset** to validate the pipeline and evaluation flow before scaling.

---

## Evaluation Metrics

The model was evaluated using standard binary classification metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score (weighted)

Additional evaluation artifacts include:
- Confusion matrix
- Class-wise performance analysis
- Misclassified example inspection

These metrics are intended to **validate model behavior**, not to claim state-of-the-art performance.

---

## Inference Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "SuganyaP/quick-distilbert-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("This movie was excellent!", return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits).item()

print("Positive" if prediction == 1 else "Negative")
