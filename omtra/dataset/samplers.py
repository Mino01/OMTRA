from torch.utils.data import Sampler, DistributedSampler


class MultiTaskSampler(Sampler):

    def __init__(self, dataset, batch_size):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
