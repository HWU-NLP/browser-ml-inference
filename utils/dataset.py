import pandas as pd
import re
from ast import literal_eval
from loguru import logger

from datasets import Dataset
import torch

import ipdb

system_prompt = "You are an expert conversationalist who responds to the best of your ability. You can identify whether or not the input message from a social media platform contains a form of gender-based violence (GBV)."

gbv_prompt = (
    lambda gbv: f"""
Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output GBV if it contains GBV, or NotGBV if not.

#### Input Text:
{gbv}

#### Answer:
""".strip()
)

class PreprocessedData:
    def __init__(
        self, 
        file, 
        task,
        instruct, 
        tokenizer, 
        max_length, 
        batch_size, 
        device,
    ) -> None:
        super(PreprocessedData, self).__init__()
        self.instruct = instruct
        self.system_prompt = system_prompt
        self.device = device
        
        self.file = file
        self.task = task
        self.label_column = "label"
        self.data, self.processed_label_column, self.labels = self.read_file() 
        self.data_size = self.data.shape[0]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.text_column = "text"
        
        self.prompt_creation()
        self.labels_encoded()
        self.input_encodings = self.calculate_encodings_input() 
        self.output_encodings = self.calculate_encodings_label() if len(self.processed_label_column) == 1 else None

        logger.info(f"\n\n******************* Dataset Stats *******************\n \
                    Total dataset: {self.data_size}\n \
                    Total unique labels: {len(self.labels)}\n \
                    ")
        
    def read_file(self):
        pass

    def labels_encoded(self): 
        self.calculate_encodings_label()

    def apply_instruction_template(self, input, output=None):   
        if self.instruct:
            self.tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
            self.tokenizer.pad_token_id = 0  # unk
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2
            
            if output is not None:
                messages = [
                    {'content': self.system_prompt,'role': 'system'},
                    {'content': input, 'role': 'user'}, 
                    {'content': output, 'role': 'assistant'},
                ] 
            else:
                messages = [
                    {'content': self.system_prompt,'role': 'system'},
                    {'content': input, 'role': 'user'}, 
                ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return input
        
    def prompt_creation(self):
        self.data['prompt'] = self.data.apply(
                lambda x: self.apply_instruction_template(
                    (gbv_prompt(x[self.text_column]))
                ), 
                axis=1
            )
        self.data['prompt_and_label'] = self.data.apply(
                lambda x: self.apply_instruction_template(
                    (gbv_prompt(x[self.text_column])),
                    x['label_text']
                ), 
                axis=1
            )
        self.data['raw_prompt'] = self.data.apply(
                lambda x: (gbv_prompt(x[self.text_column])), 
                axis=1
            )

        print('\nexample of input with prompt:\n')
        print(self.data['prompt'][10])

        print('\nexample of input with prompt and label:\n')
        print(self.data['prompt_and_label'][10])

    def calculate_encodings_input(self): 
        if self.task == "evaluate_generation":
            # for evaluate generation task, encode prompt+label text 
            encodings = self.tokenizer(
                self.input_label_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
        else:
            encodings = self.tokenizer(
                self.input_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
        return encodings

    def calculate_encodings_label(self):
        # e.g., '1 2 3' or 'gbv' 
        if self.task == "evaluate_generation":
            # for evaluate generation task, encode prompt+label text but ignore index before label
            # make input tokens to -100 so that the model doesn't compute loss on them
            input_label_encodings = self.tokenizer(
                self.input_label_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to('cuda')
            input_encodings = self.tokenizer(
                self.input_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to('cuda')
            input_label_ids = input_label_encodings["input_ids"]
            input_ids = input_encodings["input_ids"]
            label_encodings = input_label_ids.clone()
            mask = (input_label_ids == input_ids) & (input_ids != self.tokenizer.pad_token_id)
            label_encodings[mask] = -100

        else:
            label_encodings = self.tokenizer(
                self.label_text(), 
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to('cuda')
        return label_encodings

    def input_texts(self):
        return self.data["prompt"].values.tolist() 

    def label_text(self): 
        return self.data['label_text'].values.tolist()
    
    def input_label_texts(self):
        return self.data["prompt_and_label"].values.tolist()

    def __getitem__(self, idx):
        if self.task == 'generate':
            input_ids = self.input_encodings["input_ids"][idx]
            attention_mask = self.input_encodings["attention_mask"][idx]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        elif self.task == 'evaluate_generation':
            # only for a training purpose
            input_ids = self.input_encodings["input_ids"][idx]
            attention_mask = self.input_encodings["attention_mask"][idx]
            label = self.output_encodings[idx] 
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label,
            }
        else:
            raise NotImplementedError(f"Task {self.task} not implemented in __getitem__")

    def encoding_fn(self, sample):
        if type(sample) == int:
            return self.__getitem__(sample)   
        else:
            for idx in sample:
                return self.__getitem__(idx)

    def __len__(self):
        return self.data_size

    def __num_labels__(self):
        return len(self.labels)

    def __data__(self):
        return self.data

class GBV_EDOS(PreprocessedData):
    def read_file(self):
        data = pd.read_csv(self.file, sep='\t', header=0)
        label = self.label_column
        print('label column: ', label)
       
        data['label_text'] = data[label].map({"sexist": "GBV", "not sexist": "NotGBV"}).fillna(-100).tolist()
        processed_label_column = ['label_text']
        labels = data[label].unique()
        return data, processed_label_column, labels

    def tokenized_data(self):
        list_idx = []
        label_list=self.label_text()

        list_idx = self.data.index.tolist()
        texts = self.input_texts()
        dataset = {'id': list_idx, 'text': texts, 'label': label_list, 'unique_labels': self.data['text'].tolist()}
        dataset = Dataset.from_dict(dataset)
        tokenized_dataset = dataset.map(lambda example: self.encoding_fn(example["id"]))
        return tokenized_dataset  # no batched

    def splitting(self, test_size, seed=42):
        tokenized_dataset = self.tokenized_data()
        dataset_split = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)
        return dataset_split
    