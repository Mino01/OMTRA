import torch
import numpy as np
import dgl
from typing import List, Dict

from omtra.dataset.register import dataset_name_to_class
from omtra.tasks.register import task_name_to_class

class MultiMultiSet(torch.utils.data.Dataset):

    """A dataset capable of serving up samples from multiple zarr datasets."""

    def __init__(self, split: str, tasks: List[dict], dataset_paths: Dict[str, str]):
        

        dataset_names = list(dataset_paths.keys())
        dataset_classes = [dataset_name_to_class[dataset_name] for dataset_name in dataset_names]

        task_names = []
        task_probs = []
        for task_dict in tasks:
            task_names.append(task_dict['name'])
            task_probs.append(task_dict['prob'])
        task_classes = [task_name_to_class[task_name] for task_name in task_names]


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


