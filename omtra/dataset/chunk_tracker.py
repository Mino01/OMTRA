import torch

from omtra.dataset.zarr_dataset import ZarrDataset

class ChunkTracker:
    def __init__(self, dataset: ZarrDataset, *args, **kwargs):
        self.dataset = dataset
        self.graphs_per_chunk = dataset.graphs_per_chunk


        # chunk index is an (n_hunks, 2) array containing the start and end indices of each chunk
        self.chunk_index = self.dataset.retrieve_graph_chunks(*args, **kwargs)
        self.n_chunks = self.chunk_index.shape[0]

        self.chunk_queue = torch.randperm(self.n_chunks)
        self.chunk_queue_idx = 0