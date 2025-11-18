import datasets
from datasets import load_dataset, Dataset
import transformers

import os
import numpy as np
import pandas as pd
import evaluate
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from pathlib import Path

from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import train_test_split

import ipdb 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output 1 if it contains GBV, or 0 if not."
CHOICES = "Choices: 1 for GBV, or 0 for Not GBV"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def convert_hs_label(label):
    if label == "sexist":
        return 1
    elif label == "not sexist":
        return 0
    else:
        return -1

def generate_prompt(text):
    return f"{INSTRUCTION} Text: {text} {CHOICES} Answer:" 

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def dataset_loader(tokenizer):
    train_val = pd.read_csv("data/edos-train.tsv", sep="\t")
    train, val = train_test_split(train_val, test_size=0.2, random_state=SEED, shuffle=True)
    test = pd.read_csv("data/edos-test.tsv", sep="\t")
    # train["gbv_text"] = train["text"]
    # val["gbv_text"] = val["text"]
    # test["gbv_text"] = test["text"]
    # train["text"] = train["gbv_text"].apply(generate_prompt)
    # val["text"] = val["gbv_text"].apply(generate_prompt)
    # test["text"] = test["gbv_text"].apply(generate_prompt)
    
    train["label"] = train["label"].apply(convert_hs_label)
    val["label"] = val["label"].apply(convert_hs_label)
    test["label"] = test["label"].apply(convert_hs_label)
    
    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    test_dataset = Dataset.from_pandas(test.iloc[:10, :])  # reduce test size for faster debugging
    # ipdb.set_trace()
    
    tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    return tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset

def load_model(model_name: str):
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path if model_path.exists() else model_name, 
                                                               num_labels=2, 
                                                               device_map="auto",
                                                               dtype=torch.float16)
    model = model.to(DEVICE)    
    print(DEVICE)
    return model, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)
    f1score = evaluate.load("f1").compute(predictions=preds, references=labels, average='binary')['f1']
    return {"accuracy": acc["accuracy"], "f1": f1score}

def train_model(model, tokenizer, train_dataset, eval_dataset, metrics_fn):
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    training_args = TrainingArguments(
        "test_trainer_ckpt",
        per_device_train_batch_size=128, 
        num_train_epochs=3, #24,
        learning_rate=2e-05,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        seed=SEED,
    )    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_fn,
        data_collator=data_collator,
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
    model_name = 'FacebookAI/roberta-base'
    model_name_out = "onnx/gbv_classifier"
    model, tokenizer = load_model(model_name)
    train, val, test = dataset_loader(tokenizer=tokenizer)
    # ipdb.set_trace()
    
    train_model(model=model, tokenizer=tokenizer, train_dataset=train, eval_dataset=val, metrics_fn=compute_metrics)
    # ipdb.set_trace()
    
    
if __name__ == "__main__":
    main()
