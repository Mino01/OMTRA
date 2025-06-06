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
    p = argparse.ArgumentParser(description='Compute new properties for a block of Pharmit data.')

    p.add_argument('--pharmit_path', type=Path, help='Path to the Pharmit Zarr store.', default=Path('/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit'))
    p.add_argument('--block_start_idx', type=int, default=0, help='Index of the first molecule in the block.')
    p.add_argument('--block_size', type=int, default=100, help='Number of ligands to process.')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    
    args = p.parse_args()

    return args


def ligand_properties(mol: Chem.Mol) -> np.ndarray:
    """
    Parameters:
        mol (Chem.Mol): RDKit ligand

    Returns: 
        np.ndarray: Additional ligand features (n_atoms, 6)
    """

    implicit_Hs = []    # Number of implicit hydrogens (int)
    aromaticity = []    # Whether the atom is in an aromatic ring (binary flag)
    hybridization = []  # Hydridization (int)
    in_ring = []        # Whether the atom is in a ring (binary flag)
    conjugated_pi_system = []   # Whether the atom in a conjugated π system (binary flag)
    chiral_center = []         # Whether the atom is a chiral center (binary flag)

    chiral_centers = set(idx for idx, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=True))  # Collect indices of chiral atoms


    for atom in mol.GetAtoms():
        implicit_Hs.append(atom.GetNumImplicitHs())
        aromaticity.append(int(atom.GetIsAromatic()))
        hybridization.append(int(atom.GetHybridization()))
        in_ring.append(int(atom.IsInRing()))
        conjugated_pi_system.append(int(atom.GetIsConjugated()))
        chiral_center.append(int(atom.GetIdx() in chiral_centers))
    
    new_feats = np.array([
        implicit_Hs,
        aromaticity,
        hybridization,
        in_ring,
        conjugated_pi_system,
        chiral_center
    ], dtype=np.int8).T

    return new_feats


def fragment_molecule(mol: Chem.Mol) -> np.ndarray:
    """ 
    Parameters:
        mol (Chem.Mol): RDKit ligand

    Returns:
        np.ndarray: Index of the BRICS fragment for each atom (n_atoms, 1) 
    """

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

    atom_to_fragment = np.array(atom_to_fragment, dtype=np.int8)

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


def process_pharmit_block(block_start_idx: int, block_size: int):
    """ Gets the new atom properties for each molecule in the block and writes to Zarr store

    Parameters:
        pharmit_path (Path): Path to the Pharmit zarr store
        array_name (str): Name of the Zarr array to add new data to
        block_start_idx (int): Index of the first ligand in the block
        block_size (int): Number of ligands in the blocl
    """
    
    global pharmit_dataset  # Load Pharmit dataset object
    n_mols = len(pharmit_dataset)
    block_end_idx = block_start_idx + block_size


    # idx correction if pharmit dataset is not a multiple of block_size 
    if block_end_idx > n_mols: 
        block_end_idx = n_mols

    ligand_idxs = []
    new_feats = []

    for idx in range(block_start_idx, block_end_idx):
        g, start_idx, end_idx = pharmit_dataset[('denovo_ligand', idx)]   # TODO: How will PharmitDataset class be modified so we can access each molecule's atom lookup
        mol = dgl_to_rdkit(g)

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Sanitization failed for molecule {idx}: {e}.")
            # TODO: How do handle failures? In theory, this should never fail since we sanitized before storing in the Zarr store
            continue

        lig_properties = ligand_properties(mol)                         # (n_atoms, 6)
        fragments = fragment_molecule(mol)                              # (n_atoms, 1)
        lig_properties = np.concatenate((lig_properties, fragments), axis=1)  # (n_atoms, 7)

        ligand_idxs.append([start_idx, end_idx])
        new_feats.append(lig_properties)
    
    return ligand_idxs, new_feats
    

        
class BlockWriter:
    def __init__(self, pharmit_path: Path):
        self.pharmit_path = pharmit_path

    def save_chunk(self, array_name: str, ligand_idxs: np.ndarray, new_feats: np.ndarray):

        # Open Pharmit Zarr store
        root = zarr.open(self.pharmit_path, mode='r+') # read-write mode
        lig_node_group = root['lig/node']

        # Check that Zarr array was correctly made
        if array_name not in lig_node_group:
            raise KeyError(f"Zarr array '{array_name}' not found in 'lig/node' group.")

        for i, lig_properties in enumerate(new_feats):
            start_idx = ligand_idxs[i][0]
            end_idx = ligand_idxs[i][1]

            # Check that number of atoms for the ligand in the zarr store = number of atoms we have after computing new properties
            assert lig_properties.shape[0] == (end_idx - start_idx), f"Mismatch in atom counts: lig_properties has {lig_properties.shape[0]} rows but expected {(end_idx - start_idx)}"

            # write features to zarr store
            lig_node_group[array_name][start_idx:end_idx] = lig_properties



if __name__ == '__main__':
    args = parse_args()
    pharmit_path = args.pharmit_path
    array_name = args.array_name
    block_start_idx = args.block_start_idx
    block_size = args.block_size

    process_pharmit_block(pharmit_path, array_name, block_start_idx, block_size)