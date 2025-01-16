import torch
import pytorch_lightning as pl

class OMTRA(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        # implement training step
        pass

    def validation_step(self, batch, batch_idx):
        # implement validation step
        pass

    def forward(self):
        pass

    def configure_optimizers(self):
        # implement optimizer
        pass