import datasets
from datasets import load_dataset
import transformers
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

import numpy as np
import evaluate
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

def compute_metrics(eval_pred):    
    logits, labels = eval_pred    
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, train_dataset, eval_dataset, metrics_fn):
    training_args = TrainingArguments("test_trainer",
                                  per_device_train_batch_size=128, 
                                  num_train_epochs=24,
                                  learning_rate=3e-05,
                                  eval_strategy="epoch",)    
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


def predict_on_dataset(model_name, dataset, metric, quantized=True):
    
    model_name = model_name + (".onnx" if quantized else "_int8.onnx")    
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
    model_name_out = "emotion_classifier"
    dataset_name = "emotion"

    train, eval = dataset_loader(dataset_name)
    model, tokenizer = load_model(model_name)

    train_model(model=model, train_dataset=train, eval_dataset=eval, metrics_fn=compute_metrics)
    convert_to_onnx(model=model, tokenizer=tokenizer, model_name_out=model_name_out)
    metric = evaluate.load("accuracy")
    predict_on_dataset(model_name=model_name, dataset=eval, metric=metric)

if __name__ == "__main__":
    main()
