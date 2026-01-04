# =====================================================
# IMDB SLM Optimizing
# Purpose: Train & save DistilBERT sentiment model
# =====================================================

# -----------------------
# Imports
# -----------------------
import os
import time
import json
import random
import statistics
import numpy as np
import torch

from collections import Counter
from datasets import load_dataset, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------
# Paths (relative, GitHub-safe)
# -----------------------
TRAIN_CSV = "imdb_train_clean.csv"
TEST_CSV  = "imdb_test_clean.csv"

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "artifacts/quick_distilbert_model"

# -----------------------
# Load Dataset
# -----------------------
dataset = load_dataset(
    "csv",
    data_files={
        "train": TRAIN_CSV,
        "test": TEST_CSV
    }
)

if "review" not in dataset["train"].column_names or "sentiment" not in dataset["train"].column_names:
    raise ValueError("Expected columns: review, sentiment")

print("Train label distribution:", Counter(dataset["train"]["sentiment"]))

sample_texts = dataset["train"]["review"][:1000]
lengths = [len(t.split()) for t in sample_texts]
print("Median review length:", statistics.median(lengths))

# -----------------------
# Tokenization
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch["review"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized = dataset.map(tokenize_fn, batched=True)

# -----------------------
# Smoke-test Subset
# -----------------------
train_subset = tokenized["train"].select(range(min(1000, len(tokenized["train"]))))
eval_subset  = tokenized["test"].select(range(min(500, len(tokenized["test"]))))

train_subset = train_subset.rename_column("sentiment", "labels")
eval_subset  = eval_subset.rename_column("sentiment", "labels")

train_subset = train_subset.cast_column("labels", Value("int64"))
eval_subset  = eval_subset.cast_column("labels", Value("int64"))

# -----------------------
# Model
# -----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# -----------------------
# Metrics (lightweight, training-time only)
# -----------------------
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# -----------------------
# Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# -----------------------
# Train
# -----------------------
trainer.train()
print("Training complete.")

# -----------------------
# Save Artifacts
# -----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metadata = {
    "model_name": MODEL_NAME,
    "train_subset_size": len(train_subset),
    "eval_subset_size": len(eval_subset),
    "training_args": training_args.to_dict(),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Model and metadata saved to: {OUTPUT_DIR}")
