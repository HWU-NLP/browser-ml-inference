import argparse
import os
import yaml
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as pipe
import transformers.convert_graph_to_onnx as onnx_convert
from onnxruntime.quantization import quantize_dynamic, QuantType

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)    
    print(device)
    return model, tokenizer

def convert_to_onnx(model, tokenizer, model_name_out):
    pipeline = pipe("text-classification",model=model,tokenizer=tokenizer)
    model = model.to("cpu")
    onnx_convert.convert_pytorch(pipeline, opset=11, output=Path(model_name_out + ".onnx"), use_external_format=False)
    quantize_dynamic(model_name_out + ".onnx", model_name_out + "_int8.onnx", 
                 weight_type=QuantType.QUInt8)
    
def main():
    parser = argparse.ArgumentParser(description="Convert a Llama-based model into ONNX format for text classification")
    parser.add_argument("--config_file", required=True, help="Path to the config file")
    args = parser.parse_args()
    # Check if the config file exists
    assert os.path.exists(args.config_file), f"Config file '{args.config_file}' does not exist."
    
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

        model, tokenizer = load_model(model_name=config["model"])
        convert_to_onnx(model=model, tokenizer=tokenizer, model_name_out=config['model_name_out'])
    

if __name__ == "__main__":
    main()