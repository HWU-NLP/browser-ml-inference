from torch.utils.data import Sampler
from collections import defaultdict
import random
import torch


class UniqueBatchSampler(Sampler):
    def __init__(self, labels, group_keys, batch_size, random_state=42, mode="train"):
        """
        labels: list of sample identifiers (e.g. cs texts or indices)
        group_keys: list of values to group by (e.g. gbv texts or gbv ids)
        batch_size: number of unique labels per batch
        random_state: seed for reproducibility
        mode: 'train' or 'generate'
        """
        assert mode in ["train", "generate"]
        assert len(labels) == len(group_keys)

        self.labels = labels
        self.group_keys = group_keys
        self.batch_size = batch_size
        self.random_state = random_state
        self.mode = mode

        self.group_to_indices = defaultdict(list)
        for idx, group in enumerate(group_keys):
            self.group_to_indices[group].append(idx)

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
        used_indices = set()
        remaining_group_keys = self.group_keys #.copy()

        while len(used_indices) < self.total_size:
            batch = []
            available_groups = list(set([g for idx, g in enumerate(remaining_group_keys) if idx not in used_indices]))
            if not available_groups:
                break
            selected_groups = random.sample(available_groups, min(self.batch_size, len(available_groups)))
            for group in selected_groups:
                candidates = [i for i in self.group_to_indices[group] if i not in used_indices]
                if candidates:
                    idx = random.choice(candidates)
                    batch.append(idx)
                    used_indices.add(idx)
                    remaining_group_keys[idx] = None  # prevent this group from being selected again in this loop
            if batch:
                yield batch
                
                      
    def __len__(self):
        return (self.total_size + self.batch_size - 1) // self.batch_size
