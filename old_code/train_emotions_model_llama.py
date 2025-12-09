import datasets
from datasets import load_dataset
import transformers
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

import numpy as np
import evaluate
import torch
# from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import LlamaTokenizer, LlamaForSequenceClassification
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path

from transformers import TrainingArguments
from transformers import Trainer
# from transformers import export
from optimum.exporters.onnx import export

from huggingface_hub import login
hub_token = "<hf_token>"
login(token=hub_token)

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if model_path.exists() else model_name,
        use_fast=True
    )
    # tokenizer = LlamaTokenizer.from_pretrained(model_path if model_path.exists() else model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = AutoModelForSequenceClassification.from_pretrained(model_path if model_path.exists() else model_name, num_labels=6)
    model = LlamaForSequenceClassification.from_pretrained(
        model_path if model_path.exists() else model_name,
        num_labels=6,
        device_map="auto",
        # dtype=torch.float16,
        torch_dtype=torch.float16

    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
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
        "test_trainer",
    #   per_device_train_batch_size=128, 
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,  # keeps effective batch size large
        num_train_epochs=2,#3,
        learning_rate=3e-5,
        eval_strategy="epoch",
        fp16=False,
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
    trainer.save_model("test_trainer/emotion_model")

def convert_to_onnx(model, tokenizer, model_name_out, opset=18):
    pipeline = transformers.pipeline("text-classification",model=model,tokenizer=tokenizer)
    model = model.to("cpu")
    onnx_convert.convert_pytorch(
        pipeline, 
        opset=opset, 
        output=Path(model_name_out + ".onnx"), 
        use_external_format=False
    )
    # export(
    #     model=model.to("cpu"),
    #     tokenizer=tokenizer, 
    #     output=Path(model_name_out + ".onnx"),
    #     task="text-classification",
    #     opset=opset,
    # )
    quantize_dynamic(
        model_name_out + ".onnx", 
        model_name_out + "_int8.onnx", 
        weight_type=QuantType.QUInt8
    )
    print(f"ONNX and quantised models saved: {model_name_out}.onnx, {model_name_out}_int8.onnx")
    
def predict_on_dataset(model_name, model_name_out, dataset, metric, quantized=True):
    
    model_name = model_name_out + (".onnx" if quantized else "_int8.onnx")   
    session = ort.InferenceSession(model_name)    
    input_feed = {
        "input_ids": np.array(dataset['input_ids']),
        "attention_mask": np.array(dataset['attention_mask']),
        # "token_type_ids": np.array(dataset['token_type_ids'])
    }
    out = session.run(input_feed=input_feed,output_names=['output_0'])[0]
    predictions = np.argmax(out, axis=-1) # type: ignore
    return metric.compute(predictions=predictions, references=dataset['label'])

def main():
    # model_name = 'microsoft/xtremedistil-l6-h256-uncased'
    model_name = 'meta-llama/Llama-3.2-1B'
    # model_name = '/scratch/ik36/browser-ml-inference/test_trainer/checkpoint-1008'
    model_name_out = "emotion_classifier_llama"
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
