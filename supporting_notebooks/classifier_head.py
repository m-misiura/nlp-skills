# %%
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# %% load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("../trained_models/bert-tiny-llm-router")
model = AutoModel.from_pretrained("../trained_models/bert-tiny-llm-router")

# %%
input_text = "The cat sat on the mat"
input_tokens = tokenizer(input_text, return_tensors="pt")
outputs = model(input_tokens["input_ids"])
hidden_states = (
    outputs.last_hidden_state
)  # Shape: [batch_size, sequence_length, hidden_size]

# %%
# Option 1: Use [CLS] token (first token) for classification
cls_token_state = hidden_states[:, 0, :]  # Shape: [batch_size, hidden_size]

# Option 2: Pool all token representations
mean_pooled = torch.mean(hidden_states, dim=1)  # Shape: [batch_size, hidden_size]
max_pooled = torch.max(hidden_states, dim=1)[0]  # Shape: [batch_size, hidden_size]


# Define a simple classifier head
class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


# Initialize and use the classifier
num_labels = 2  # binary classification
classifier = ClassifierHead(hidden_size=128, num_labels=num_labels)
logits = classifier(cls_token_state)

# %%
classifier = ClassifierHead(hidden_size=128, num_labels=2)

# Calculate total parameters
total_params = sum(p.numel() for p in classifier.parameters())
weight_params = 128 * 2  # hidden_size * num_labels
bias_params = 2  # num_labels

print(f"Total parameters: {total_params}")
print(f"Weight parameters: {weight_params}")
print(f"Bias parameters: {bias_params}")
print(
    f"Parameter breakdown: {128}×{2} weights + {2} bias = {weight_params + bias_params}"
)
# %%
print(f"Logits shape: {logits.shape}")
print(f"Logits: {logits}")

# %%
# Convert logits to probabilities using softmax
probabilities = F.softmax(logits, dim=-1)
print(f"Probabilities: {probabilities}")
print(f"Sum of probabilities: {probabilities.sum().item()}")  # Should be 1.0
# %%
