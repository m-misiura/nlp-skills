# %%
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from random import randrange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torchinfo import summary
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view, head_view
import os
import torch
from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments

# %% login to hf
load_dotenv()
TOKEN = os.getenv("hf_token_write")
login(token=TOKEN, add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# %% load dataset
# Dataset id from huggingface.co/dataset
dataset_id = "DevQuasar/llm_router_dataset-synth"

# Load raw dataset
raw_dataset = load_dataset(dataset_id)

print(f"Train dataset size: {len(raw_dataset['train'])}")
print(f"Test dataset size: {len(raw_dataset['test'])}")

# %% check an example
random_id = randrange(len(raw_dataset["train"]))
raw_dataset["train"][random_id]

# %% load tokenizer
# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = (
    512  # set model_max_length to 512 as prompts are not longer than 1024 tokens
)


# %% tokenize dataset
# Tokenize helper function
def tokenize(batch):
    return tokenizer(
        batch["prompt"], padding="max_length", truncation=True, return_tensors="pt"
    )


# Tokenize dataset
# raw_dataset = raw_dataset.rename_column("label", "labels")  # to match Trainer
tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["prompt"])

print(tokenized_dataset["train"].features.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask','lable'])

# %% check what tokenized dataset looks like
print(tokenized_dataset["train"][random_id])
print(tokenized_dataset["train"]["input_ids"][random_id])
print(tokenized_dataset["train"]["attention_mask"][random_id])
# decode tokenized example
tokenizer.decode(tokenized_dataset["train"][random_id]["input_ids"])

# %% load model
# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"

# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["label"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

# %% view config
model.config

# %% get more info into model vocab
# get vocab size
vocab_size = model.config.vocab_size
print(f"Vocab size: {vocab_size}")

# %% print examples from a vocab
# print first 100 tokens
for i in range(100):
    print(f"Token: {tokenizer.decode(i)}, ID: {i}")

# %% print examples from a vocab
# print special tokens
print("\nSpecial tokens and their IDs:")
for token_type, token in tokenizer.special_tokens_map.items():
    if isinstance(token, str):
        print(f"{token_type}: '{token}' (ID: {tokenizer.convert_tokens_to_ids(token)})")
    else:
        print(f"{token_type}:")
        for t in token:
            print(f"  '{t}' (ID: {tokenizer.convert_tokens_to_ids(t)})")


# %% print examples from a vocab
# print alphabetical tokens
vocab = tokenizer.get_vocab()  # Get
print("Single alphabet characters:")
for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if char in vocab:
        print(f"Token: '{char}', ID: {vocab[char]}")
    else:
        # Some tokenizers might have different representations
        print(f"Character '{char}' not found as a direct token")

# %% print examples from a vocab
# see if word yes is in vocab
word = "yes"
print(f"\nMethod 1: Direct vocabulary lookup:")
for candidate in [word]:  # Check different tokenizer formats
    if candidate in vocab:
        print(f"'{candidate}' found in vocabulary with ID: {vocab[candidate]}")
    else:
        print(f"'{candidate}' not found directly in vocabulary")


# %% visualise model architecture
# Create sample inputs matching ModernBERT expectations
sample_input = {
    "input_ids": torch.ones(1, 512, dtype=torch.long),
    "attention_mask": torch.ones(1, 512, dtype=torch.long),
}
summary(model, input_data=sample_input, device="cpu", verbose=1)

# %% show some visualisatins
utils.logging.set_verbosity_error()
model_viz = AutoModel.from_pretrained(model_id, output_attentions=True)

# Test input
input_text = "The cat sat on the mat"
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Get model outputs with attention
outputs = model_viz(inputs)
attention = outputs.attentions

# Get tokens for visualization
tokens = tokenizer.convert_ids_to_tokens(inputs[0])
# Visualize attention patterns
model_view(attention, tokens)

# %% head view
head_view(attention, tokens)

# %% check out the input embeddings
input_text = "The cat sat on the mat"
input_tokens = tokenizer(input_text, return_tensors="pt")
input_embeddings = model.get_input_embeddings()
embeddings = input_embeddings(input_tokens["input_ids"])

# Display the actual embeddings and their shape
print(f"Input text: '{input_text}'")
print(f"Tokenized into: {len(tokens)} tokens")
print(
    f"Embedding shape: {embeddings.shape} (batch_size × sequence_length × embedding_dim)"
)
print("\nEmbedding statistics per token:")
for i, token in enumerate(tokens):
    emb = embeddings[0, i].detach()
    print(
        f"Token {i}: '{token}' | Norm: {torch.norm(emb):.2f} | Min: {emb.min():.2f} | Max: {emb.max():.2f} | Mean: {emb.mean():.2f}"
    )

# %% get an embedding for the first token
first_token_embedding = embeddings[0, 0]
first_token_embedding

# %% get all hidden states
outputs = model_viz(input_tokens["input_ids"], output_hidden_states=True)
hidden_states = outputs.hidden_states


# %%
print("\nTransformation of [CLS] token through layers:")
for layer_idx, layer_state in enumerate(hidden_states):
    cls_state = layer_state[0, 0]
    print(
        f"Layer {layer_idx}: Norm={torch.norm(cls_state):.2f}, Mean={cls_state.mean():.2f}"
    )

# %% examine the classification head
input_text = "The cat sat on the mat"
inputs = tokenizer(input_text, return_tensors="pt")

# Step 1: Get the outputs from the classification model
classification_outputs = model(**inputs)
logits = classification_outputs.logits
print(f"Logits shape: {logits.shape}")  # [batch_size, num_classes]
print(f"Raw logits: {logits}")

# %%
# Step 2: Convert to probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)
print(f"\nProbabilities: {probabilities}")

# %%
# Step 3: Get the predicted class
predicted_class_id = logits.argmax(-1).item()
predicted_class = id2label[str(predicted_class_id)]
print(f"\nPredicted class: {predicted_class} (ID: {predicted_class_id})")
print(f"Confidence: {probabilities[0][predicted_class_id].item():.4f}")

# %%
# Step 4: Examine all class probabilities
print("\nAll class probabilities:")
for i, label in id2label.items():
    print(f"{label}: {probabilities[0][int(i)].item():.4f}")

# Step 5: Explore the classifier architecture
print("\nClassifier architecture:")
# The classifier is typically the last module in the model
classifier = model.classifier if hasattr(model, "classifier") else model.score

# Print classifier parameters and shape
print(f"Classifier type: {type(classifier)}")
print(f"Classifier parameters:")
for name, param in classifier.named_parameters():
    print(f"  {name}: {param.shape}")

total_params = sum(p.numel() for p in classifier.parameters())
print(f"\nTotal classifier parameters: {total_params}")
print(
    f"Input features: {classifier.in_features if hasattr(classifier, 'in_features') else 'N/A'}"
)
print(
    f"Output classes: {classifier.out_features if hasattr(classifier, 'out_features') else num_labels}"
)

# %% fine tune the model
import numpy as np
from sklearn.metrics import f1_score


# Metric helper method
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
    output_dir="modernbert-llm-router",
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
# %%
trainer.train()

# %%
