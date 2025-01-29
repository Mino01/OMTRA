import hydra

import pytorch_lightning as pl
from omtra.dataset.data_module import MultiTaskDataModule
from pathlib import Path
import omtra

from omegaconf import DictConfig, OmegaConf


def omtra_root_resolver():
    return str(Path(omtra.__file__).parent.parent)

OmegaConf.register_new_resolver("omtra_root", omtra_root_resolver, replace=True)

def train(cfg: DictConfig):
    """Trains the model.

    cfg is a DictConfig configuration composed by Hydra.
    """
    # set seed everywhere (pytorch, numpy, python)
    pl.seed_everything(cfg.seed, workers=True)

    print(f"⚛ Instantiating datamodule <{cfg.task_group.data._target_}>")
    datamodule: MultiTaskDataModule = hydra.utils.instantiate(cfg.task_group.data)

    datamodule.setup(stage='fit')
    # TODO: Implmenet other instantiation logic


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for training.

    cfg is a DictConfig configuration composed by Hydra.
    """
    print("\n=== Training Config ===")
    print(OmegaConf.to_yaml(cfg))

    # train the model
    _ = train(cfg)

if __name__ == "__main__":
    main()