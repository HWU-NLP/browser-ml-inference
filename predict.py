import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time
import json

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

def generate_prompt(text):
    return f"{INSTRUCTION} Text: {text} {CHOICES} Answer:" 

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding='max_length')

def dataset_loader(tokenizer):
    test = pd.read_csv("data/edos-test.tsv", sep="\t")
    test["label"] = test["label"].map({"sexist": 1, "not sexist": 0}).fillna(-100)
    test["gbv_text"] = test["text"]
    test["text"] = test["gbv_text"].apply(generate_prompt)
    
    test_dataset = Dataset.from_pandas(test) #.iloc[:100, :])  # reduce test size for faster debugging
    tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    ipdb.set_trace()
    return tokenized_test_dataset

def load_model(model_name: str):
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

def load_llm_model(model_name: str):
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else model_name)
    
    # Some LLM tokenizers (e.g. Qwen variants) don't define a pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path if model_path.exists() else model_name,
        num_labels=2,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))
    try:
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            model.config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    model = model.to(DEVICE)
    print(DEVICE)
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id, "model.config.pad_token_id:", getattr(model.config, "pad_token_id", None))
    return model, tokenizer

def load_json(fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "r") as fp:
        return json.load(fp)
    
def predict_on_batch_dataset(model, tokenizer, dataset, batch_size=32):
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            input_ids = torch.Tensor(batch["input_ids"]).long().to(DEVICE)
            attention_mask = torch.Tensor(batch["attention_mask"]).long().to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

            all_preds.extend(preds)
            batch_labels = batch["label"]
            all_labels.extend([int(x) for x in batch_labels])

    # Filter out examples with ignored label (-100)
    filtered_preds = []
    filtered_refs = []
    for p, r in zip(all_preds, all_labels):
        if r == -100:
            continue
        filtered_preds.append(int(p))
        filtered_refs.append(int(r))

    acc = evaluate.load("accuracy").compute(predictions=filtered_preds, references=filtered_refs)
    f1score = evaluate.load("f1").compute(predictions=filtered_preds, references=filtered_refs, average='binary')['f1']
    return {"accuracy": f"{acc['accuracy']:.4f}", "f1": f"{f1score:.4f}"}

def predict_onnx_on_dataset(model_name, model_name_out, dataset, quantized=True):
    model_name = model_name_out + (".onnx" if not quantized else "_int8.onnx")    
    session = ort.InferenceSession(model_name)   
     
    input_feed = {
        "input_ids": np.array(dataset['input_ids']),
        "attention_mask": np.array(dataset['attention_mask']),
    }
    outputs = session.run(input_feed=input_feed,output_names=['output_0'])[0]
    predictions = np.argmax(outputs, axis=-1) 
    return metric.compute(predictions=filtered_preds, references=filtered_refs)

def predict_onnx_on_batch_dataset(model_name, model_name_out, dataset, quantized=True, batch_size=32):
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

    return evaluate.load("accuracy").compute(predictions=all_preds, references=all_labels)

def predict_on_llm_result(dataset):
    filtered_preds = []
    filtered_refs = []
    
    for item in dataset:
        pred = item["prediction"].strip().lower()
        label = item["label"].strip().lower()
        if label == "gbv":
            label_int = 1
        elif label == "notgbv":
            label_int = 0
        else:
            continue
        if pred == "gbv":
            pred_int = 1
        elif pred == "notgbv":
            pred_int = 0
        else:
            continue
        filtered_preds.append(pred_int)
        filtered_refs.append(label_int)
    
    acc = evaluate.load("accuracy").compute(predictions=filtered_preds, references=filtered_refs)
    f1score = evaluate.load("f1").compute(predictions=filtered_preds, references=filtered_refs, average='binary')['f1']
    return {"accuracy": f"{acc['accuracy']:.4f}", "f1": f"{f1score:.4f}"}

def main():
    start_time = time.time()
    
    # # prediction on bert model
    # model_name = 'Heriot-WattUniversity/gbv-classifier-roberta-base-instruct'
    # model, tokenizer = load_model(model_name)
    # test = dataset_loader(tokenizer=tokenizer)
    # result = predict_on_batch_dataset(model=model, tokenizer=tokenizer, dataset=test, batch_size=32)
    # print(f"GBV model result: {result} -- based on {model_name}")
    
    # # prediction on llm model
    # model_name = '/home/aj2066/browser-ml-inference/test_trainer/gbv_qwen2'
    # model, tokenizer = load_llm_model(model_name)
    # test = dataset_loader(tokenizer=tokenizer)
    # result = predict_on_batch_dataset(model=model, tokenizer=tokenizer, dataset=test, batch_size=32)
    # print(f"GBV model result: {result} -- based on {model_name}")
    
    # prediction on llm model
    # predict_fp = '/home/aj2066/browser-ml-inference/outputs/gbv_SmolLM2-135M-Instruct_gbv_generation_outputs.json'
    predict_fp = '/home/aj2066/browser-ml-inference/outputs/gbv_Qwen2.5-0.5B-Instruct_gbv_generation_outputs.json'
    model_name = predict_fp.split("/")[-1].split("_gbv_generation_outputs.json")[0]
    outputs = load_json(predict_fp)
    predictions = pd.DataFrame(outputs)['prediction']
    print(f"Label distribution in predictions:\n{predictions.value_counts()}")
    result = predict_on_llm_result(outputs)
    print(f"GBV model result: {result} -- based on {model_name}")
    
    # # prediction on onnx model
    # model_name = 'FacebookAI/roberta-base'
    # model_name_out = "onnx/gbv_classifier"
    # model, tokenizer = load_model(model_name)
    # test = dataset_loader(tokenizer=tokenizer)
    # # result = predict_on_dataset(model_name=model_name, model_name_out=model_name_out, dataset=test, metric=metric)
    # result = predict_on_batch_dataset(model_name=model_name, model_name_out=model_name_out, dataset=test, metric=metric, batch_size=32)
    # print(f"ONNX gbv model accuracy: {result} -- based on {model_name}")
    
    end_time = time.time()
    print(f"Total prediction time: {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()

'''
Fro wolpertinger
CUDA_VISIBLE_DEVICES=1 uv run python predict.py
'''