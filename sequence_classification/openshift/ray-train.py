import os
import shutil
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
import transformers
def train_func():
    dataset_id = "DevQuasar/llm_router_dataset-synth"
    raw_dataset = load_dataset(dataset_id)
    model_id = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512

    def tokenize(batch):
        return tokenizer(
            batch["prompt"], padding="max_length", truncation=True, return_tensors="pt"
        )

    tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["prompt"])

    # Prepare model labels
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

    # Freeze base model layers, train classifier only
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # Metric helper
    def compute_metrics(eval_pred):
        predictions, labels_ = eval_pred
        predictions = np.argmax(predictions, axis=1)
        score = f1_score(labels_, predictions, average="weighted")
        return {"f1": float(score)}

    # Training arguments
    training_args = TrainingArguments(
        output_dir="bert-tiny-llm-router-ray",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        # gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        dataloader_num_workers=8,
        gradient_checkpointing=False,
        num_train_epochs=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        disable_tqdm=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    # Ray integration (optional, only if ray.train.huggingface.transformers is available)
    try:
        import ray.train.huggingface.transformers
        callback = ray.train.huggingface.transformers.RayTrainReportCallback()
        trainer.add_callback(callback)
        trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    except ImportError:
        pass

    trainer.train()
    trainer.evaluate()

# --- Ray distributed launch ---
if __name__ == "__main__":
    ray.init()
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        run_config=RunConfig(storage_path="s3://ray-trial-s3/ray_results")
    )
    result = ray_trainer.fit()
    
    