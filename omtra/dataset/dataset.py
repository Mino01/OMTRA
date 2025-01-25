import torch
import zarr
import numpy as np
import zarr.storage
import dgl

from abc import ABC, abstractmethod

class MultiMultiSet(torch.utils.data.Dataset):

    """A dataset capable of serving up samples from multiple zarr datasets."""

    def __init__(self, split: str):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class BaseOmtraDataset(ABC, torch.utils.data.Dataset):

    def __init__(self, zarr_store_path: str):
        super().__init__()

        self.store = zarr.storage.LocalStore(zarr_store_path)
        self.root = zarr.open(store=self.store, mode='r')


        # by assuming some default structure to our zarr data, we can directly infer the number of examples in the dataset
        # actually this is bad - this may be constricting, easier to just let __len__ be abstract and have the user implement it
        # example_group_name = list(root.groups())[0][0]
        # example_group = root[example_group_name]
        # try:
        #     lookup_key = [ array_name for array_name in example_group.array_keys() if 'lookup' in array_name][0]
        # except IndexError:
        #     raise ValueError('Zarr groups must have instance-specific lookup arrays with "lookup" in the name')
        # n_examples = example_group[lookup_key].shape[0]
        

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx) -> dgl.DGLHeteroGraph:
        pass


class PharmitDataset(BaseOmtraDataset):

    def __init__(self, zarr_store_path: str):
        super().__init__(zarr_store_path)

    def __len__(self):
        return self.root['node_data']['node_lookup'].shape[0]

    def __getitem__(self, idx) -> dgl.DGLHeteroGraph:
        pass