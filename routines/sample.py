# predict.py
import os
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
import omtra.load.quick as quick_load
import torch

from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class

from omtra.utils import omtra_root
from pathlib import Path
OmegaConf.register_new_resolver("omtra_root", omtra_root, replace=True)

default_config_path = Path(omtra_root()) / 'configs'
default_config_path = str(default_config_path)

@hydra.main(config_path=default_config_path, config_name="sample")
def main(cfg):
    # 1) resolve checkpoint path
    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = OmegaConf.load(train_cfg_path)
    
    # override anything in the training config file with anything passed in (sample.yaml by default)
    # or anything passed in via the command line, i.e., if you need to override the pharmit or plinder paths
    merged_cfg = OmegaConf.merge(train_cfg, cfg)

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    dm  = quick_load.datamodule_from_config(cfg)
    multitask_dataset = dm.load_dataset('val')
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device)
    
    # get task we are sampling for
    task_name: str = cfg.task
    task: Task = task_name_to_class(task_name)

    # get raw dataset object
    if cfg.dataset == 'plinder':
        plinder_link_version = task.plinder_link_version
        dataset = multitask_dataset.datasets['plinder'][plinder_link_version]
    elif cfg.dataset == 'pharmit':
        dataset = multitask_dataset.datasets['pharmit']
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

    # get g_list
    if task.unconditional:
        g_list = None
        n_replicates = cfg.n_samples
    else:
        dataset_idxs = range(cfg.dataset_start_idx, cfg.dataset_start_idx + cfg.n_samples)
        dataset_entires = [ dataset[(task_name, i)] for i in dataset_idxs ]
        g_list, _, _ = zip(*dataset_entires)
        g_list = list(g_list)
        g_list = [ g.to(device) for g in g_list ] # move to device
        n_replicates = cfg.n_replicates

    model.sample(
        g_list=g_list,
        n_replicates=n_replicates,
        task_name=task_name,
        dataset_name=cfg.dataset,
        unconditional_n_atoms_dist=cfg.dataset,
        device=device,
        n_timesteps=cfg.n_timesteps,
    )
        

if __name__ == "__main__":
    main()