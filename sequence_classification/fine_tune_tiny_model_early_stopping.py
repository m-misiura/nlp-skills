# %%
import os
import torch
import shutil
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv
from random import randrange
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torchinfo import summary

# %% load the raw dataset
dataset_id = "DevQuasar/llm_router_dataset-synth"
raw_dataset = load_dataset(dataset_id)

# Split train into train/validation for early stopping & hyperparam search
split = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
    "test": raw_dataset["test"]
})

# Load model tokenizer
model_id = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512

# %% Tokenize dataset
def tokenize(batch):
    return tokenizer(
        batch["prompt"], padding="max_length", truncation=True, return_tensors="pt"
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt"])

# %% load model
labels = tokenized_dataset["train"].features["label"].names
num_labels = len(labels)
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

# %% # Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
        labels, predictions, labels=labels, pos_label=1, average="weighted"
    )
    return {"f1": float(score) if score == 1 else score}

# %% Training arguments with Adafactor optimizer
training_args = TrainingArguments(
    output_dir="bert-tiny-llm-router",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    optim="adafactor",  # Use Adafactor optimizer
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    disable_tqdm=False,
    report_to="none", 
)

# %% Freeze base model layers
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

def count_trainable_parameters(model):
    trainable = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(
        f"trainable params: {trainable:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable / all_param:.2f}%"
    )
    return trainable

trainable_params = count_trainable_parameters(model)

# %% Trainer with early stopping
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# %% Hyperparameter search using Optuna
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        # "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
        # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8]),
    }

best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=10)
print("Best run:", best_run)

# %% Train with best hyperparameters
for key, value in best_run.hyperparameters.items():
    setattr(trainer.args, key, value)
trainer.train()

# %% Evaluate on test set
trainer.evaluate(tokenized_dataset["test"])

# %% Save model and tokenizer
trainer.save_model("../trained_models/bert-tiny-llm-router")
tokenizer.save_pretrained("../trained_models/bert-tiny-llm-router")

# %% Remove the temporary model directory
if os.path.exists("./bert-tiny-llm-router"):
    shutil.rmtree("./bert-tiny-llm-router")

# %% Push to Hugging Face Hub
trainer.push_to_hub(commit_message="Add fine-tuned BERT-tiny for LLM routing")