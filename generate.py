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

import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.dataset import GBV_EDOS, PreprocessedData
from utils.sampler import UniqueBatchSampler
import ipdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 4

# Debug mode: set `DEBUG_TRAIN=1` in environment to run a fast, low-memory end-to-end
DEBUG = bool(int(os.environ.get("DEBUG_TRAIN", "0"))) or os.environ.get("DEBUG", "0") == "1"

MODEL_NAME = "/home/aj2066/browser-ml-inference/test_trainer/gbv_SmolLM2-135M-Instruct"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(model_name: str):
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path if model_path.exists() else model_name,
        device_map="auto",
    )
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
    sampler = UniqueBatchSampler(
        labels=inputs["label"],
        group_keys=inputs["unique_labels"], 
        batch_size=batch_size, 
        random_state=random_state,
        mode="generate") 
   
    decoded_outputs = []
    n = 0
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
        decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for idx, output in zip(batch["id"], decoded_output):
            decoded_outputs.append({'text': output,
                                    'prompt': output.split('Answer:')[0], 
                                    'gbv': dataset.data['text'][idx],
                                    'output': output.split('Answer:')[1], 
                                    'id': int(dataset.data['id'][idx]),
                                    'label': int(dataset.data['label_text'][idx])})
          
    logger.warning(f"Number of outputs with errors: {n}\n")
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
    test_data_path = "data/edos-test.tsv"
    logger.info(f"**** Loading dataset from {test_data_path}\n")
    dataset = GBV_EDOS(
        file=test_data_path, 
        task="generate",  
        tokenizer=tokenizer,
        max_length=1024,
        batch_size=BATCH_SIZE,
        instruct = True,
        device=DEVICE,
    )
    logger.debug(dataset)
    ipdb.set_trace()
    
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
        max_new_tokens=1, 
        do_sample=True, 
        temperature=0.6,
        top_p=0.9,
        pad_token_id=dataset.tokenizer.pad_token_id, 
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
    logger.info(f"Time cost: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()

"""
For wolpertinger:
CUDA_VISIBLE_DEVICES=1 uv run python generate.py 

For debugging:
DEBUG_TRAIN=1 CUDA_VISIBLE_DEVICES=1 uv run python generate.py
"""