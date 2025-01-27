import torch
from torch.utils.data import Sampler, DistributedSampler


class MultiTaskSampler(Sampler):

    def __init__(self, dataset, batch_size):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        

        self.task_names = dataset.task_names
        self.dataset_names = dataset.dataset_names
        self.p_dataset_task = dataset.p_dataset_task

    def sample_task_and_dataset(self):

        p = self.p_dataset_task

        # Flatten the tensor to work with torch.multinomial
        flat_p = p.flatten()

        # Draw a single sample from the flattened distribution
        index = torch.multinomial(flat_p, 1).item()

        # Convert the flat index back to 2D indices
        n, m = p.shape
        task_idx, dataset_idx = divmod(index, m)

        return task_idx, dataset_idx


    def __iter__(self):

        # TODO: (i'm working on it i promise)
        pass