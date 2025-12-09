import os
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from loguru import logger
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from transformers import Trainer, TrainingArguments

from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

from utils.dataset import GBV_EDOS
from utils.sampler import UniqueBatchSampler

from huggingface_hub import login
import ipdb 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["WANDB_PROJECT"] = "gbv_classification" 
os.environ["WANDB_API_KEY"] = "<wandb_api_key>"

hub_token = "<hf_token>"
login(token=hub_token)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 4
EPOCHS = 5

# Debug mode: set `DEBUG_TRAIN=1` in environment to run a fast, low-memory end-to-end
DEBUG = bool(int(os.environ.get("DEBUG_TRAIN", "0"))) or os.environ.get("DEBUG", "0") == "1"

MODEL_NAME = 'HuggingFaceTB/SmolLM2-135M-Instruct' #'Qwen/Qwen2.5-0.5B-Instruct'
print(DEVICE)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

class DataCollatorForPrompt:
    def __init__(self, tokenizer):
        self.collator = DataCollatorWithPadding(tokenizer, padding=True)  # padding="max_length",

    def __call__(self, batch):
        # Only keep model inputs
        cleaned_batch = [{k: v for k, v in d.items() if k in ['input_ids', 'attention_mask', 'labels']} for d in batch]
        return self.collator(cleaned_batch)

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, random_state=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state
        
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=UniqueBatchSampler(
                labels=[s['label'] for s in self.train_dataset],
                group_keys=[s['unique_labels'] for s in self.train_dataset],
                batch_size=self.args.per_device_train_batch_size,
                random_state=self.random_state,
            ),
            collate_fn=DataCollatorForPrompt(self.tokenizer),  
            num_workers=2
        )
        
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_sampler=UniqueBatchSampler(
                labels=[s['label'] for s in self.eval_dataset],
                group_keys=[s['unique_labels'] for s in self.eval_dataset],
                batch_size=self.args.per_device_eval_batch_size,
                random_state=self.random_state,
            ),
            collate_fn=DataCollatorForPrompt(self.tokenizer), 
            num_workers=2
        )

def load_model(model_name: str):
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_path if model_path.exists() else model_name,
        device_map="auto",
    )
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # model = get_peft_model(model, config)
    
    model = model.to(DEVICE)
    return model, tokenizer
    
def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir, wandb_run_name): 
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.0
    )
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.0

    training_args = TrainingArguments(
        "test_trainer_ckpt",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        num_train_epochs=(1 if DEBUG else EPOCHS), 
        learning_rate=2e-5,
        # lr_scheduler_type = "linear",
        weight_decay=0.001,
        warmup_ratio=0.05,
        optim="adamw_torch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=SEED,
        report_to=["wandb"],
        run_name=f"{MODEL_NAME.split('/')[-1]}-gbv-classifier",
        fp16=True,
        max_grad_norm=1.0,
    )    

    data_collator = DataCollatorForPrompt(tokenizer)
    
    trainer = Trainer(
        model=model,
        tokenizer = tokenizer,  
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_cb],
    )

    trainer.train()

    logger.info(f"**** Saving finetuned model to {output_dir}\n")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
    
def main():
    start_time = time.time()
        
    set_seed(SEED)
    dash_line = "\n" + "_".join(" " for x in range(100))

    logger.info(dash_line)
    logger.info(f"**** Loading model: {MODEL_NAME}\n")
    model, tokenizer = load_model(MODEL_NAME)

    logger.info(dash_line)
    train_data_path = "data/edos-train.tsv"
    logger.info(f"**** Loading dataset from {train_data_path}\n")
    dataset = GBV_EDOS(
        file=train_data_path, 
        task="evaluate_generation",  # "generate" for inference
        tokenizer=tokenizer,
        max_length=1024,
        batch_size=BATCH_SIZE,
        instruct = True,
        device=DEVICE,
    )
    split_dataset = dataset.splitting(test_size=0.2, seed=SEED)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    logger.debug(train_dataset)
    logger.debug(eval_dataset)
    
    # If debug mode, use very small subsets to run quickly
    if DEBUG:
        try:
            if hasattr(train_dataset, "select"):
                train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
            if hasattr(eval_dataset, "select"):
                eval_dataset = eval_dataset.select(range(min(20, len(eval_dataset))))
        except Exception:
            pass
    
    logger.info(dash_line)
    logger.info(f"**** Starting fine-tuning\n")
    output_dir = f"test_trainer/gbv_{MODEL_NAME.split('/')[-1]}"
    wandb_run_name=f"{MODEL_NAME.split('/')[-1]}-gbv-classifier"
    
    model.config.pad_token_id = dataset.tokenizer.pad_token_id 
    model.config.bos_token_id = dataset.tokenizer.bos_token_id
    model.config.eos_token_id = dataset.tokenizer.eos_token_id
    logger.info(print_number_of_trainable_model_parameters(model))
    
    trainer = train_model(
        model=model, 
        tokenizer=dataset.tokenizer, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        output_dir=output_dir,
        wandb_run_name=wandb_run_name,
    )

    ipdb.set_trace()
    
    end_time = time.time()
    logger.info(f"Time cost: {end_time - start_time:.4f} seconds")

    
if __name__ == "__main__":
    main()

'''
For wolpertinger
CUDA_VISIBLE_DEVICES=1 uv run python train_gbv_llm.py

For debugging:
DEBUG_TRAIN=1 CUDA_VISIBLE_DEVICES=1 uv run python train_gbv_llm.py
'''