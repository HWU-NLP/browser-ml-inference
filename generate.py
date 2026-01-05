import os
from pathlib import Path
from typing import Any, Dict
import yaml
import time
import pandas as pd
from tqdm import tqdm
from loguru import logger
import random
import numpy as np
import json
import re

import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.dataset import GBV_EDOS, PreprocessedData
from utils.sampler import BatchSampler
import ipdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 4

# Debug mode: set `DEBUG_TRAIN=1` in environment 
DEBUG = bool(int(os.environ.get("DEBUG_TRAIN", "0"))) or os.environ.get("DEBUG", "0") == "1"

TEST_DATA_PATH = "data/edos-test.tsv"
# MODEL_NAME = "/home/aj2066/browser-ml-inference/test_trainer/gbv_SmolLM2-135M-Instruct"
MODEL_NAME = "/home/aj2066/browser-ml-inference/test_trainer/gbv_Qwen2.5-0.5B-Instruct"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(model_name: str):
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path if model_path.exists() else model_name,
        device_map="auto",
    )
    try:
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            model.config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass
    
    return model, tokenizer

def generate_from_dataset_batch(
    dataset: PreprocessedData, 
    model: Any, 
    tokenizer: Any, 
    params: GenerationConfig,
    batch_size: int,
    random_state: int,
    device: torch.device
    ) -> None:

    inputs = dataset.tokenized_data()
    print('\n--> sampling')
    sampler = BatchSampler(
        labels=inputs["label"],
        batch_size=batch_size, 
        random_state=random_state,
        mode="generate") 
   
    decoded_outputs = []
    for batch_indices in tqdm(sampler):
        batch = inputs.select(batch_indices)

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=params,
            )
        gen_only_ids = output_ids[:, input_ids.shape[1]:]
        decoded_gen = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
        decoded_full = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for idx, output_full, output_gen in zip(batch["id"], decoded_full, decoded_gen):
            print(f"\nDEBUG output for id {idx}:\n{output_gen}\n")
            suffix = (output_gen or "").strip()
            if not suffix:
                m = re.split(r"####\s*Answer:\s*|\bAnswer:\s*", output_full, flags=re.I)
                suffix = (m[-1] if m else output_full).strip()

            suffix = re.sub(r"^\s*(assistant[:\s\n]+|assistant\s*)", "", suffix, flags=re.I)
            first_line = suffix.splitlines()[0].strip() if suffix else ""
            first_line = re.sub(r"^[\W_]+", "", first_line)

            # Normalise predictions into GBV / NotGBV labels
            s_up = first_line.upper()
            if 'NOTGBV' in s_up:
                pred_label = 'NotGBV'
            elif 'GBV' in s_up:
                pred_label = 'GBV'
            elif re.search(r'\bNOT\b|\bNO\b|\bNOT\b', s_up):
                pred_label = 'NotGBV'
            elif re.search(r'\bGBV\b|\bYES\b|\bY\b', s_up):
                pred_label = 'GBV'
            else:
                pred_label = 'NotGBV'
            print(f"DEBUG pred_label for id {idx}: {pred_label}")
            
            decoded_outputs.append({'text': output_full,
                                    'prompt': output_full,
                                    'gbv': dataset.data['text'][idx],
                                    'output': (output_gen or ""),
                                    'prediction': pred_label,
                                    'id': dataset.data['id'][idx],
                                    'label': dataset.data['label_text'][idx]})
    return decoded_outputs

def save_json(data: Any, fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "w") as fp:
        return json.dump(data, fp, indent=4)

def main():
    start_time = time.time()
    
    set_seed(seed=SEED)
    dash_line = "\n" + "_".join(" " for x in range(100))

    logger.info(dash_line)
    logger.info(f"**** Loading model: {MODEL_NAME}\n")
    model, tokenizer = load_model(MODEL_NAME)

    logger.info(dash_line)
    logger.info(f"**** Loading dataset from {TEST_DATA_PATH}\n")
    dataset = GBV_EDOS(
        file=TEST_DATA_PATH, 
        task="generate",  
        tokenizer=tokenizer,
        max_length=1024,
        batch_size=BATCH_SIZE,
        instruct = True,
        device=DEVICE,
    )
    logger.debug(dataset)
    # ipdb.set_trace()
    
    # If debug mode, use very small subsets to run quickly
    if DEBUG:
        try:
            if hasattr(dataset, "select"):
                dataset = dataset.select(range(min(20, len(dataset))))
        except Exception:
            pass
    
    model.config.pad_token_id = dataset.tokenizer.pad_token_id 
    model.config.bos_token_id = dataset.tokenizer.bos_token_id
    model.config.eos_token_id = dataset.tokenizer.eos_token_id

    generation_params = GenerationConfig(
        max_new_tokens=3,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=dataset.tokenizer.pad_token_id, 
        eos_token_id=dataset.tokenizer.eos_token_id,
    ) 
        
    logger.info(dash_line)
    logger.info(f"**** Generating outputs\n")

    outputs = generate_from_dataset_batch(
        dataset=dataset, 
        model=model,
        tokenizer=dataset.tokenizer,
        params=generation_params,
        batch_size=BATCH_SIZE,
        random_state=SEED,
        device=DEVICE,
    )
    
    fpath = Path("outputs", f"{MODEL_NAME.split('/')[-1]}_gbv_generation_outputs.json")
    save_json(data=outputs, fpath=str(fpath))     
    logger.success(f"Generated {len(outputs)} outputs are save to {fpath}")

    end_time = time.time()
    logger.info(f"\nTime cost: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()

"""
For wolpertinger:
CUDA_VISIBLE_DEVICES=1 uv run python generate.py 

For debugging:
DEBUG_TRAIN=1 CUDA_VISIBLE_DEVICES=1 uv run python generate.py
"""