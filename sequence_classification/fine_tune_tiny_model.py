# %%
import os
import torch
import shutil
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from random import randrange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torchinfo import summary
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view, head_view
from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments

# %% load the raw dataset
# Dataset id from huggingface.co/dataset
dataset_id = "DevQuasar/llm_router_dataset-synth"

# Load raw dataset
raw_dataset = load_dataset(dataset_id)

# Load model tokenizer
model_id = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = (
    512  # set model_max_length to 512 as prompts are not longer than 1024 tokens
)


# %% Tokenize dataset
# raw_dataset = raw_dataset.rename_column("label", "labels")  # to match Trainer
def tokenize(batch):
    return tokenizer(
        batch["prompt"], padding="max_length", truncation=True, return_tensors="pt"
    )


tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["prompt"])

# %% load model
# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["label"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


# Download the model
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


# %%
# Define training args
training_args = TrainingArguments(
    output_dir="bert-tiny-llm-router",
    # Memory-efficient batch settings
    per_device_train_batch_size=2,  # Small batch size
    per_device_eval_batch_size=2,  # Small evaluation batch size
    gradient_accumulation_steps=16,  # Accumulate gradients to simulate larger batch
    # Optimizer settings
    learning_rate=5e-5,
    warmup_ratio=0.1,  # Gradual warmup to stabilize training
    optim="adamw_torch",  # Standard optimizer works best with MPS
    # Memory optimizations
    fp16=False,  # MPS doesn't support fp16 well
    bf16=False,  # No bfloat16 support on MPS
    dataloader_num_workers=0,  # Avoid memory overhead from workers
    gradient_checkpointing=True,  # Trade computation for memory
    # Regular training parameters
    num_train_epochs=5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # Keep only last 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    # Display settings
    disable_tqdm=False,  # Enable
)

# %% Train the head only
print("Model structure:")
for name, _ in model.named_parameters():
    print(name)

# Freeze base model layers (modernbert instead of bert)
for name, param in model.named_parameters():
    if "classifier" not in name:  # Keep classifier trainable
        param.requires_grad = False


# Verify trainable parameters
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


# Check trainable parameters
trainable_params = count_trainable_parameters(model)

# %%
# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
# %% train the model
trainer.train()

# %% evaluate the model
trainer.evaluate()

# %% save the model
# Save the model to the output directory
trainer.save_model("../trained_models/bert-tiny-llm-router")

# %% save the tokenizer
tokenizer.save_pretrained("../trained_models/bert-tiny-llm-router")

# %% remove the temporary model directory
if os.path.exists("./bert-tiny-llm-router"):
    shutil.rmtree("./bert-tiny-llm-router")

# %% push to Hugging Face Hub
trainer.push_to_hub(commit_message="Add fine-tuned BERT-tiny for LLM routing")
