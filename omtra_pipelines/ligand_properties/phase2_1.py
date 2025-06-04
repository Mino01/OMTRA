import argparse
from typing import List, Dict
from pathlib import Path
import numpy as np
import dgl
import zarr

from rdkit import Chem
from rdkit.Chem import BRICS

from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load
from omtra.tasks.register import task_name_to_class
from omtra.eval.system import SampledSystem


def parse_args():
    p = argparse.ArgumentParser(description='Compute new properties for a block of pharmit data.')

    p.add_argument('--pharmit_path', type=Path, help='Path to the Pharmit Zarr store.', default=Path('/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit'))
    p.add_argument('--block_start_idx', type=int, default=0, help='Index of the first molecule in the block.')
    p.add_argument('--block_size', type=int, default=3, help='Number of ligands to process in the block.')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    
    args = p.parse_args()

    return args


def ligand_properties(mol: Chem.Mol) -> Dict[str, List]:
    """ Gets aromaticity and number of implicit hydrogens for each atom in a RDKit ligand returned as dictionary """

    implicit_Hs = []
    aromaticity = []
    
    for atom in mol.GetAtoms():
        implicit_Hs.append(atom.GetNumImplicitHs())
        aromaticity.append(atom.GetIsAromatic())
        
    aromaticity = [int(a) for a in aromaticity]
    return np.array([implicit_Hs, aromaticity]).T


def fragment_molecule(mol: Chem.Mol) -> List:
    """ Gets the BRICS fragment assignment for each atom """

    broken = BRICS.BreakBRICSBonds(mol) # cut molecule at BRICS bonds and replace with dummy atoms labeled [*]

    # TODO: Check for errors in fragment generation

    # find connected components
    comps = Chem.GetMolFrags(broken, asMols=False)     # returns tuple of tuples. each tuple is a connected component

    # build mapping from each original atom to fragment
    N = mol.GetNumAtoms()
    atom_to_fragment = [-1] * N

    for frag_idx, comp in enumerate(comps):
        for ai in comp:
            atom = broken.GetAtomWithIdx(ai)
            if atom.GetSymbol() != "*" and ai < N: # not part of a BRICS bond
                atom_to_fragment[ai] = frag_idx
    atom_to_fragment = np.array(atom_to_fragment)
    return atom_to_fragment[:, np.newaxis]


def move_feats_to_t1(task_name: str, g: dgl.DGLHeteroGraph, t: str = '0'):
    task = task_name_to_class(task_name)
    for m in task.modalities_present:

        num_entries = g.num_nodes(m.entity_name) if m.is_node else g.num_edges(m.entity_name)
        if num_entries == 0:
            continue

        data_src = g.nodes if m.is_node else g.edges
        dk = m.data_key
        en = m.entity_name

        if t == '0' and m in task.modalities_fixed:
            data_to_copy = data_src[en].data[f'{dk}_1_true']
        else:
            data_to_copy = data_src[en].data[f'{dk}_{t}']

        data_src[en].data[f'{dk}_1'] = data_to_copy

    return g


def dgl_to_rdkit(g):
    """ Converts one DGL molecule to RDKit ligand """

    g = move_feats_to_t1('denovo_ligand', g, '1_true')
    rdkit_ligand = SampledSystem(g).get_rdkit_ligand()
    return rdkit_ligand


def process_pharmit_block(pharmit_path: Path, array_name: str, block_start_idx: int, block_size: int):
    """ Gets the new atom properties for each molecule in the block and writes to Zarr store """

    # Load Pharmit dataset object
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("val")
    pharmit_dataset = train_dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    block_end_idx = block_start_idx + block_size

    # Open Pharmit Zarr store
    root = zarr.open(pharmit_path, mode='r+') # read-write mode
    lig_node_group = root['lig/node']

    # Check that Zarr array was correctly made
    if array_name not in lig_node_group:
        raise KeyError(f"Zarr array '{array_name}' not found in 'lig/node' group.")


    # idx correction if pharmit dataset is not a multiple of block_size 
    if block_end_idx > n_mols: 
        block_end_idx = n_mols

    for idx in range(block_start_idx, block_end_idx):
        g, start_idx, end_idx = pharmit_dataset[('denovo_ligand', idx)]   # TODO: How will PharmitDataset class be modified so we can access each molecule's atom lookup
        mol = dgl_to_rdkit(g)

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Sanitization failed for molecule {idx}: {e}.")
            # TODO: How do handle failures?
            continue

        lig_properties = ligand_properties(mol)                         # (n_atoms, 2)
        fragments = fragment_molecule(mol)                              # (n_atoms, 1)
        lig_properties = np.concatenate((lig_properties, fragments), axis=1)  # (n_atoms, 3)
        lig_properties = lig_properties.astype(np.int8)

        # write features to zarr store
        assert lig_properties.shape[0] == (end_idx - start_idx), f"Mismatch in atom counts: lig_properties has {lig_properties.shape[0]} rows but expected {(end_idx - start_idx)}"
        lig_node_group[array_name][start_idx:end_idx] = lig_properties


if __name__ == '__main__':
    args = parse_args()
    pharmit_path = args.pharmit_path
    array_name = args.array_name
    block_start_idx = args.block_start_idx
    block_size = args.block_size

    process_pharmit_block(pharmit_path, array_name, block_start_idx, block_size)