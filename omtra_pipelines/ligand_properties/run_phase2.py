import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time

from multiprocessing import Pool
from functools import partial

from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load

from omtra_pipelines.ligand_properties.phase2 import *


def parse_args():
    p = argparse.ArgumentParser(description='Generate embarassingly parallel processing commands for phase2_1.py')

    p.add_argument('--pharmit_path', type=Path, help='Path to the Pharmit Zarr store.', default=Path('/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit'))
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--block_size', type=int, default=10000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=2, help='Number of CPUs to use for parallel processing.')

    return p.parse_args()


def worker_initializer(pharmit_path):
    global pharmit_dataset
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("val")
    pharmit_dataset = train_dataset.datasets['pharmit']
    
    
def save_and_update(block_writer, array_name, ligand_idxs, new_feats, pbar):
    block_writer.save_chunk(array_name, ligand_idxs, new_feats)
    # Update the progress bar by one step.
    pbar.update(1)

def error_and_update(error, pbar, error_counter):
    """Handle errors, update error counter and the progress bar."""
    print(f"Error: {error}")
    traceback.print_exception(type(error), error, error.__traceback__)
    # Increment the error counter (using a mutable container)
    error_counter[0] += 1
    # Optionally, update the tqdm bar's postfix to show the current error count.
    pbar.set_postfix({'errors': error_counter[0]})
    # Advance the progress bar, since this job is considered done.
    pbar.update(1)

    # write the traceback to a file named 'error_log.txt'
    with open('error_log.txt', 'a') as f:
        traceback.print_exception(type(error), error, error.__traceback__, file=f)


def run_parallel(pharmit_path: Path,
                 array_name: str,
                 block_size: int,
                 n_cpus: int,
                 block_writer: BlockWriter,
                 max_pending: int = None):
    
    # Set a default limit if not provided
    if max_pending is None:
        max_pending = n_cpus * 2  # adjust this factor as needed

    # Load Pharmit dataset
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("val")
    pharmit_dataset = train_dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)

    # Calculate number of blocks needed to cover entire dataset
    n_blocks = (n_mols + block_size - 1) // block_size

    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="chunks")
    error_counter = [0] # Use a mutable container to track errors.

    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(pharmit_path,), maxtasksperchild=2) as pool:
        pending = []

        for block_idx in range(n_blocks):
            while len(pending) >= max_pending:
                # Filter out jobs that have finished
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)  # brief pause before checking again


            callback_fn = partial(save_and_update, block_writer=block_writer, array_name=array_name, pbar=pbar)
            error_callback_fn = partial(error_and_update, pbar=pbar, error_counter=error_counter)

            # Submit the job and add its AsyncResult to the pending list
            result = pool.apply_async(
                process_pharmit_block, 
                args=(block_idx * block_size, block_size), 
                callback=callback_fn,
                error_callback=error_callback_fn
            )
            pending.append(result)

        # After submitting all jobs, wait for any remaining tasks to complete.
        for result in pending:
            result.wait()

        pool.close()
        pool.join()
        
def main():
    args = parse_args()
    pharmit_path = args.pharmit_path

    block_writer = BlockWriter(pharmit_path)

    start_time = time.time()
    run_parallel(pharmit_path, args.array_name, args.block_size, args.n_cpus)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")
    

if __name__ == '__main__':
    main()


