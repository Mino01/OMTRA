import argparse
from pathlib import Path

from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load


def parse_args():
    p = argparse.ArgumentParser(description='Generate parallel processing commands for phase2_1.py')

    p.add_argument('--pharmit_path', type=Path, help='Path to the Pharmit Zarr store.', default=Path('/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit'))
    p.add_argument('--block_size', type=int, default=10000, help='Number of ligands to process in the block.')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--output_file', type=Path, default=Path('process_pharmit_cmds.txt'), help='Output text file with commands.')

    return p.parse_args()


def main():
    args = parse_args()

    # Load Pharmit dataset
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=args.pharmit_path)
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("val")
    pharmit_dataset = train_dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    block_size = args.block_size

    # Calculate number of blocks needed to cover entire dataset
    n_blocks = (n_mols + block_size - 1) // block_size

    with open(args.output_file, 'w') as f:
        for block_idx in range(n_blocks):
            cmd = (
                f"python phase2_1.py "
                f"--pharmit_path {args.pharmit_path} "
                f"--block_start_idx {block_idx * block_size} "
                f"--block_size {block_size} "
                f"--array_name {args.array_name}"
            )
            f.write(cmd + '\n')

if __name__ == '__main__':
    main()


