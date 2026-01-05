from torch.utils.data import Sampler
from collections import defaultdict
import random
import torch


class BatchSampler(Sampler):
    def __init__(self, labels, batch_size, random_state=42, mode="train"):
        """
        labels: list of sample identifiers (e.g. cs texts or indices)
        batch_size: number of unique labels per batch
        random_state: seed for reproducibility
        mode: 'train' or 'generate'
        """
        assert mode in ["train", "generate"]
        # assert len(labels) == len(group_keys)

        self.labels = labels
        # self.group_keys = group_keys
        self.batch_size = batch_size
        self.random_state = random_state
        self.mode = mode

        # self.group_to_indices = defaultdict(list)
        # for idx, group in enumerate(group_keys):
        #     self.group_to_indices[group].append(idx)

        self.total_size = len(labels)
        
    def __iter__(self):
        if self.mode == "train":
            return self._train_iter()
        else:
            return self._generate_iter()

    def _train_iter(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.random_state if worker_info is None else self.random_state + worker_info.id
        random.seed(worker_seed)
        return self._batch_iterator()

    def _generate_iter(self):
        random.seed(self.random_state)
        return self._batch_iterator()
    
    def _batch_iterator(self):
        indices = list(range(self.total_size))
        random.shuffle(indices)
        for start in range(0, self.total_size, self.batch_size):
            yield indices[start : start + self.batch_size]
                
                      
    def __len__(self):
        return (self.total_size + self.batch_size - 1) // self.batch_size
