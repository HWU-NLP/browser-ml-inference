import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import datasets
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import evaluate
import onnxruntime as ort

import ipdb

INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output 1 if it contains GBV, or 0 if not."
CHOICES = "Choices: 1 for GBV, or 0 for Not GBV"
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    test_dataset = Dataset.from_pandas(test.iloc[:50, :])  # reduce test size for faster debugging
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

def predict_on_dataset(model_name, model_name_out, dataset, metric, quantized=True):
    model_name = model_name_out + (".onnx" if not quantized else "_int8.onnx")    
    session = ort.InferenceSession(model_name)   
     
    input_feed = {
        "input_ids": np.array(dataset['input_ids']),
        "attention_mask": np.array(dataset['attention_mask']),\
    }
    outputs = session.run(input_feed=input_feed,output_names=['output_0'])[0]
    predictions = np.argmax(outputs, axis=-1) 
    
    return metric.compute(predictions=predictions, references=dataset['label'])

def predict_on_batch_dataset(model_name, model_name_out, dataset, metric, quantized=True, batch_size=32):
    model_name = model_name_out + (".onnx" if not quantized else "_int8.onnx")    
    session = ort.InferenceSession(model_name)  
    
    all_preds, all_labels = [], []
    input_ids = np.array(dataset["input_ids"])
    attention_mask = np.array(dataset["attention_mask"])
    labels = np.array(dataset["label"])
     
    for i in tqdm(range(0, len(input_ids), batch_size), desc="Predicting"):
        input_feed = {
            "input_ids": input_ids[i:i+batch_size],
            "attention_mask": attention_mask[i:i+batch_size],
        }

        outputs = session.run(output_names=["output_0"], input_feed=input_feed)[0]
        preds = np.argmax(outputs, axis=-1)
        
        all_preds.extend(preds)
        all_labels.extend(labels[i:i+batch_size])

    return metric.compute(predictions=all_preds, references=all_labels)
 
def main():
    model_name = 'FacebookAI/roberta-base'
    model_name_out = "onnx/gbv_classifier"
    dataset_name = "dair-ai/emotion" #"emotion"
    model, tokenizer = load_model(model_name)
    _, _, test = dataset_loader(dataset_name, tokenizer=tokenizer)
    
    metric = evaluate.load("accuracy")
    # result = predict_on_dataset(model_name=model_name, model_name_out=model_name_out, dataset=test, metric=metric)
    result = predict_on_batch_dataset(model_name=model_name, model_name_out=model_name_out, dataset=test, metric=metric, batch_size=32)
    print(f"ONNX gbv model accuracy: {result} -- based on {model_name}")
    
if __name__ == "__main__":
    main()