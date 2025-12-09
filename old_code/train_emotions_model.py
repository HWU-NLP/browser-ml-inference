import datasets
from datasets import load_dataset
import transformers
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

import os
import numpy as np
import evaluate
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path

from transformers import TrainingArguments
from transformers import Trainer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def dataset_loader(dataset_name: str, tokenizer):
    dataset = load_dataset(dataset_name)
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    return full_train_dataset, full_eval_dataset

def load_model(model_name: str):
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_path if model_path.exists() else model_name, num_labels=6)
    model = model.to(device)    
    print(device)
    return model, tokenizer

def compute_metrics(eval_pred):    
    logits, labels = eval_pred    
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, train_dataset, eval_dataset, metrics_fn):
    training_args = TrainingArguments(
        "test_trainer1",
        per_device_train_batch_size=128, 
        num_train_epochs=2, #24,
        learning_rate=3e-05,
        eval_strategy="epoch",
        # save_strategy="epoch",
        # save_total_limit=2,
        # load_best_model_at_end=True,
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
    trainer.save_model("test_trainer1/emotion_model")

def convert_to_onnx(model, tokenizer, model_name_out, opset=18):
    pipeline = transformers.pipeline("text-classification",model=model,tokenizer=tokenizer)
    model = model.to("cpu")
    onnx_convert.convert_pytorch(
        pipeline, 
        opset=opset, 
        output=Path(model_name_out + ".onnx"), 
        use_external_format=False
    )
    quantize_dynamic(
        model_name_out + ".onnx", 
        model_name_out + "_int8.onnx", 
        weight_type=QuantType.QUInt8,
    )
    print(f"ONNX and quantised models saved: {model_name_out}.onnx, {model_name_out}_int8.onnx")



def predict_on_dataset(model_name, model_name_out, dataset, metric, quantized=True):
    
    model_name = model_name_out + (".onnx" if quantized else "_int8.onnx")    
    session = ort.InferenceSession(model_name)    
    input_feed = {
        "input_ids": np.array(dataset['input_ids']),
        "attention_mask": np.array(dataset['attention_mask']),
        "token_type_ids": np.array(dataset['token_type_ids'])
    }
    out = session.run(input_feed=input_feed,output_names=['output_0'])[0]
    predictions = np.argmax(out, axis=-1) # type: ignore
    return metric.compute(predictions=predictions, references=dataset['label'])
    
def main():
    model_name = 'microsoft/xtremedistil-l6-h256-uncased'
    model_name_out = "onnx/emotion_classifier1"
    dataset_name = "dair-ai/emotion" #"emotion"
    model, tokenizer = load_model(model_name)
    train, eval = dataset_loader(dataset_name, tokenizer=tokenizer)
    train_model(model=model, train_dataset=train, eval_dataset=eval, metrics_fn=compute_metrics)
    
    opset = 20
    convert_to_onnx(model=model, tokenizer=tokenizer, model_name_out=model_name_out, opset=opset)
    
    metric = evaluate.load("accuracy")
    result = predict_on_dataset(model_name=model_name, model_name_out=model_name_out, dataset=eval, metric=metric)
    print(f"ONNX model accuracy: {result} -- based on {model_name}")
    
if __name__ == "__main__":
    main()
