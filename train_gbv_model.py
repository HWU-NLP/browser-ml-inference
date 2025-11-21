import os
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
# from transformers.integrations import NeptuneCallback
from transformers import Trainer, TrainingArguments

import datasets
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import evaluate

import ipdb 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

wandb_run_name = "roberta-base-gbv-classifier"
os.environ["WANDB_PROJECT"] = "gbv_classification" 
os.environ["WANDB_API_KEY"] = "4416c892140d26422657ac3d4e51faf82783c630"

INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output 1 if it contains GBV, or 0 if not."
CHOICES = "Choices: 1 for GBV, or 0 for Not GBV"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
print(DEVICE)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def generate_prompt(text):
    return f"{INSTRUCTION} Text: {text} {CHOICES} Answer:" 

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def dataset_loader(tokenizer):
    train_val = pd.read_csv("data/edos-train.tsv", sep="\t")
    train_val["label"] = train_val["label"].map({"sexist": 1, "not sexist": 0}).fillna(-100)
    train, val = train_test_split(train_val, test_size=0.2, random_state=SEED, shuffle=True, stratify=train_val["label"])
    test = pd.read_csv("data/edos-test.tsv", sep="\t")
    test["label"] = test["label"].map({"sexist": 1, "not sexist": 0}).fillna(-100)
    # train["gbv_text"] = train["text"]
    # val["gbv_text"] = val["text"]
    # test["gbv_text"] = test["text"]
    # train["text"] = train["gbv_text"].apply(generate_prompt)
    # val["text"] = val["gbv_text"].apply(generate_prompt)
    # test["text"] = test["gbv_text"].apply(generate_prompt)
    
    train = pd.concat([train_val, test], ignore_index=True)
    
    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    test_dataset = Dataset.from_pandas(test.iloc[:10, :])  # reduce test size for faster debugging
    # ipdb.set_trace()
    
    tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    # ipdb.set_trace()
    
    return tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset

def compute_class_weights(dataset):
    label_counts = Counter(dataset["label"])
    num_neg = label_counts.get(0, 0)   # Not GBV
    num_pos = label_counts.get(1, 0)   # GBV

    # Avoid division by zero; if one class is missing, return uniform weights
    if num_neg == 0 or num_pos == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)

    total = float(num_neg + num_pos)
    weight_neg = total / (2.0 * float(num_neg))
    weight_pos = total / (2.0 * float(num_pos))

    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(DEVICE)
    return class_weights

def load_model(model_name: str, class_weights=None):
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path if model_path.exists() else model_name,
        num_labels=2,
        device_map="auto",
    )
    model = model.to(DEVICE)    
    print(DEVICE)
    return model, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)
    f1score = evaluate.load("f1").compute(predictions=preds, references=labels, average='binary')['f1']
    return {"accuracy": acc["accuracy"], "f1": f1score}

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.labels
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        class_weights = compute_class_weights(self.train_dataset)
        loss = torch.nn.functional.cross_entropy(
                logits, 
                labels, 
                weight=class_weights,
                ignore_index=-100,
            )
        return (loss, outputs) if return_outputs else loss
    
def train_model(model, tokenizer, train_dataset, eval_dataset, metrics_fn):
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0
    )
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1

    training_args = TrainingArguments(
        "test_trainer_ckpt",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=20, 
        learning_rate=2e-5,
        weight_decay=0.001,
        warmup_ratio=0.05,
        optim="adamw_torch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        seed=SEED,
        report_to=["wandb"],
        run_name=wandb_run_name,
        max_grad_norm=1.0,
        save_total_limit=3,
    )    
    
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_fn,
        data_collator=data_collator,
        callbacks=[early_stopping_cb],
    )
    trainer.train()
    trainer.evaluate()
    
    label2id = {"Not GBV": 0, "GBV": 1}
    id2label = {v: k for k, v in label2id.items()}
    trainer.model.config.label2id = label2id
    trainer.model.config.id2label = id2label

    trainer.save_model("test_trainer/gbv_model")
    tokenizer.save_pretrained("test_trainer/gbv_model")
    
def main():
    set_seed(SEED)
    model_name = 'FacebookAI/roberta-base'
    model, tokenizer = load_model(model_name)
    train, val, test = dataset_loader(tokenizer=tokenizer)
    
    train_model(model=model, tokenizer=tokenizer, train_dataset=train, eval_dataset=val, metrics_fn=compute_metrics)
    # ipdb.set_trace()
    
if __name__ == "__main__":
    main()

'''
Fro wolpertinger
CUDA_VISIBLE_DEVICES=1 uv run python train_gbv_model.py
'''