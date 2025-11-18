from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers.convert_graph_to_onnx as onnx_convert
from onnxruntime.quantization import quantize_dynamic, QuantType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def convert_to_onnx(model, tokenizer, model_name_out, opset=18):
    model = model.to("cpu").to(torch.float32)
    pipeline = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
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

def main():
    model_dir = "test_trainer/gbv_model"  # 'Heriot-WattUniversity/gbv-classifier-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    
    convert_to_onnx(
        model=model, 
        tokenizer=tokenizer, 
        model_name_out="onnx/gbv_classifier_roberta_base", 
        opset=20
    )
    

    model = model.to(DEVICE)
    model.eval()

    # test output logits
    inputs = tokenizer("I hate you!", return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    print('Has logits?', hasattr(outputs, 'logits'))
    print('Logits shape:', outputs.logits.shape)
    print(outputs.logits)

if __name__ == "__main__":
    main()