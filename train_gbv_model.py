import datasets
from datasets import load_dataset, Dataset
import transformers
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

import os
import numpy as np
import pandas as pd
import evaluate
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path

from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import train_test_split

import ipdb 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output A if it contains GBV, or B if not."
CHOICES = "Choices: 1 for GBV, or 0 for Not GBV"

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

def dataset_loader(dataset_name: str, tokenizer):
    # dataset = load_dataset(dataset_name)
    train_val = pd.read_csv("data/edos-train.tsv", sep="\t")
    train, val = train_test_split(train_val, test_size=0.2, random_state=42, shuffle=True)
    test = pd.read_csv("data/edos-test.tsv", sep="\t")
    train["gbv_text"] = train["text"]
    val["gbv_text"] = val["text"]
    test["gbv_text"] = test["text"]
    train["text"] = train["gbv_text"].apply(generate_prompt)
    val["text"] = val["gbv_text"].apply(generate_prompt)
    test["text"] = test["gbv_text"].apply(generate_prompt)
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_path if model_path.exists() else model_name, 
                                                               num_labels=2, 
                                                               device_map="auto",
                                                               dtype=torch.float16)
    model = model.to(device)    
    print(device)
    return model, tokenizer

def compute_metrics(eval_pred):    
    logits, labels = eval_pred    
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, tokenizer, train_dataset, eval_dataset, metrics_fn):
    training_args = TrainingArguments(
        "test_trainer_ckpt",
        per_device_train_batch_size=128, 
        num_train_epochs=10, #24,
        learning_rate=2e-05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_fn,
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model("test_trainer/gbv_model")
    tokenizer.save_pretrained("test_trainer/gbv_model")

def convert_to_onnx(model, tokenizer, model_name_out, opset=18):
    model = model.to("cpu").to(torch.float32)
    pipeline = transformers.pipeline("text-classification",model=model,tokenizer=tokenizer)
    onnx_convert.convert_pytorch(
        pipeline, 
        opset=opset, 
        output=Path(model_name_out + ".onnx"), 
        use_external_format=False
    )
    quantize_dynamic(
        model_name_out + ".onnx", 
        model_name_out + "_int8.onnx", 
        weight_type=QuantType.QUInt8
    )
    print(f"ONNX and quantised models saved: {model_name_out}.onnx, {model_name_out}_int8.onnx")

def predict_on_dataset(model_name, model_name_out, dataset, metric, quantized=True):
    model_name = model_name_out + (".onnx" if not quantized else "_int8.onnx")    
    session = ort.InferenceSession(model_name)    
    input_feed = {
        "input_ids": np.array(dataset['input_ids']),
        "attention_mask": np.array(dataset['attention_mask']),
    }
    out = session.run(input_feed=input_feed,output_names=['output_0'])[0]
    predictions = np.argmax(out, axis=-1) 
    return metric.compute(predictions=predictions, references=dataset['label'])
    
def main():
    # model_name = 'microsoft/xtremedistil-l6-h256-uncased'
    model_name = 'FacebookAI/roberta-base'
    model_name_out = "onnx/gbv_classifier"
    dataset_name = "dair-ai/emotion" #"emotion"
    model, tokenizer = load_model(model_name)
    train, val, test = dataset_loader(dataset_name, tokenizer=tokenizer)
    # ipdb.set_trace()
    
    train_model(model=model, tokenizer=tokenizer, train_dataset=train, eval_dataset=val, metrics_fn=compute_metrics)
    # ipdb.set_trace()
    
    opset = 20
    convert_to_onnx(model=model, tokenizer=tokenizer, model_name_out=model_name_out, opset=opset)
    
    metric = evaluate.load("accuracy")
    result = predict_on_dataset(model_name=model_name, model_name_out=model_name_out, dataset=test, metric=metric)
    print(f"ONNX gbv model accuracy: {result} -- based on {model_name}")
    
if __name__ == "__main__":
    main()
