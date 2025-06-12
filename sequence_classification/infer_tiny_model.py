# %% module imports
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# %% use high level pipeline api to get a prediction
classifier = pipeline(
    "text-classification", model="../trained_models/bert-tiny-llm-router", device=0
)

sample = "How does the structure and function of plasmodesmata affect cell-to-cell communication and signaling in plant tissues, particularly in response to environmental stresses?"


pred = classifier(sample)
print(pred)

# %% load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("../trained_models/bert-tiny-llm-router")
inputs = tokenizer(sample, return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained(
    "../trained_models/bert-tiny-llm-router"
)

# %% Step 1: Get the outputs from the classification model
classification_outputs = model(**inputs)
logits = classification_outputs.logits
print(f"Logits shape: {logits.shape}")  # [batch_size, num_classes]
print(f"Raw logits: {logits}")

# %% Step 2: Convert to probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)
print(f"\nProbabilities: {probabilities}")

# %% # Step 3: Get the predicted class and its score
# Get the model's config and extract the id2label mapping
model_config = model.config
id2label = model_config.id2label

# After getting your predictions
predicted_class_id = logits.argmax(-1).item()
predicted_class_score = probabilities[0, predicted_class_id].item()
predicted_label = id2label[predicted_class_id]

print(f"\nPredicted class: {predicted_label} (ID: {predicted_class_id})")
print(f"Confidence: {predicted_class_score:.4f}")

# To format it like the pipeline output
formatted_result = [{"label": predicted_label, "score": predicted_class_score}]
print(f"\nFormatted result: {formatted_result}")

# %%
