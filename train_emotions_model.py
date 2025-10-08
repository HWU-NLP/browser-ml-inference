import datasets
from datasets import load_dataset
import transformers
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path

from transformers import TrainingArguments
from transformers import Trainer

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def dataset_loader(dataset_name: str):
    dataset = load_dataset(dataset_name)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    return full_train_dataset, full_eval_dataset
    

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    model = model.to(device)    
    print(device)
    return model, tokenizer

def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, train_dataset, eval_dataset, metrics_fn):
    training_args = TrainingArguments("test_trainer",
                                  per_device_train_batch_size=128, 
                                  num_train_epochs=24,
                                  learning_rate=3e-05)    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_fn,
    )
    trainer.train()
    trainer.evaluate()

def convert_to_onnx(model, tokenizer, model_name_out):
    pipeline = transformers.pipeline("text-classification",model=model,tokenizer=tokenizer)
    model = model.to("cpu")
    onnx_convert.convert_pytorch(pipeline, opset=11, output=Path(model_name_out + ".onnx"), use_external_format=False)
    quantize_dynamic(model_name_out + ".onnx", model_name_out + "_int8.onnx", 
                 weight_type=QuantType.QUInt8)


def predict_on_dataset(model_name, dataset, quantized=True):
    if quantized:
        model_name = model_name + ".onnx"
    else:
        model_name = model_name + "_int8.onnx"
    session = ort.InferenceSession(model_name)    

def main():
    model_name = 'microsoft/xtremedistil-l6-h256-uncased'
    dataset_name = "emotion"

    train, eval = dataset_loader(dataset_name)
    model, tokenizer = load_model(model_name)

    train_model(model=model, train_dataset=train, eval_dataset=eval, metrics_fn=compute_metrics)

if __name__ == "__main__":
    main()
