import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import pymysql
import itertools
import re

from rdkit.Chem import AllChem as Chem
import numpy as np
import os
from multiprocessing import Pool
import pickle
from functools import partial


from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.graph import build_lookup_table
from omtra.data.pharmit_pharmacophores import get_lig_only_pharmacophore
from tempfile import TemporaryDirectory
import time


class NameFinder():

    def __init__(self, spoof_db=False):

        if spoof_db:
            self.conn = None
        else:
            self.conn = pymysql.connect(
                host="localhost",
                user="pharmit", 
                db="conformers",)
                # password="",
                # unix_socket="/var/run/mysqld/mysqld.sock")

    def query_name(self, smiles: str):

        if self.conn is None:
            return ['PubChem', 'ZINC', 'MolPort']

        with self.conn.cursor() as cursor:
            cursor.execute("SELECT name FROM names WHERE smile = %s", (smiles,))
            names = cursor.fetchall()
        names = list(itertools.chain.from_iterable(names))
        return self.extract_prefixes(names)
    

    def query_name_batch(self, smiles_list: list[str]):
        if self.conn is None:
            return [['PubChem', 'ZINC', 'MolPort'] for smiles in smiles_list], []

        # Ensure the input is unique to avoid unnecessary duplicates in the result -> Need to preserve list order to match to other lists. TODO: find smarter way to handle duplicates
        #smiles_list = list(set(smiles_list))

        with self.conn.cursor() as cursor:
            # Use the IN clause to query multiple SMILES strings
            query = "SELECT smile, name FROM names WHERE smile IN %s"
            cursor.execute(query, (tuple(smiles_list),))
            results = cursor.fetchall()

        # Organize results into a dictionary: {smile: [names]}
        smiles_to_names = {smile: None for smile in smiles_list}
        
        for smile, name in results:
            if smile not in smiles_to_names:
                smiles_to_names[smile] = []
            smiles_to_names[smile].append(name)
        
        failed_idxs = [smiles_list.index(smile) for smile in smiles_to_names if smiles_to_names[smile] is None]  # Get the indices of failed smile in smiles_list
        names = [names for smile, names in smiles_to_names.items() if names is not None] # Remove None entries 

        return names, failed_idxs
    

    def extract_prefixes(self, names):
        """
        Extracts prefixes from a list of names where the prefix consists of 
        all characters at the start of the string that are not numbers or special characters.
        
        Args:
            names (list of str): A list of strings representing molecule names.
            
        Returns:
            list of str: A list of prefixes extracted from the names.
        """
        prefixes = set()
        for name in names:
            # Use a regex to match all letters at the start of the string
            match = re.match(r'^[A-Za-z]+', name)
            if match:
                prefixes.add(match.group(0))
            else:
                continue
        return list(prefixes)
    
    def query_smiles_from_file(self, conformer_file: Path):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT smile FROM structures WHERE sdfloc = %s", (str(conformer_file),))
            smiles = cursor.fetchall()
        return smiles


    def query_smiles_from_file_batch(self, conformer_files: list[Path]):

        if self.conn is None:
            return ['CC' for file in conformer_files], []

        file_to_smile = {Path(file): None for file in conformer_files}  # Dictionary to map conformer file to smile

        # failure will be different than a mysqlerror; there just wont be an entry if the file is not in the database
        with self.conn.cursor() as cursor:
            query = "SELECT sdfloc, smile FROM structures WHERE sdfloc IN %s"
            cursor.execute(query, [tuple(str(file) for file in conformer_files)])
            results = cursor.fetchall()

        for sdfloc, smile in results:
            file_to_smile[Path(sdfloc)] = smile  # Update with successfull queries

        failed_idxs = []
        for i, file in enumerate(conformer_files):
            if file_to_smile[Path(file)] is None:
                failed_idxs.append(i)
        
        smiles = [smile for smile in file_to_smile.values() if smile is not None] # Remove None entries 

        return smiles, failed_idxs
    

    def query_names_from_files_batch(self, filepaths: list[Path]):
        """
        Given a list of filepaths, return the names linked to each of those filepaths
        with a single query to the database.

        Args:
            filepaths (list of Path): A list of filepaths to query.

        Returns:
            list of list of str: A list of lists, where each sublist contains the names
                                 associated with the corresponding filepath.
            list of int: A list of indices for filepaths that failed to retrieve names.
        """
        if self.conn is None:
            return [['PubChem', 'ZINC', 'MolPort'] for _ in filepaths], []

        file_to_names = {Path(file): None for file in filepaths}  # Dictionary to map file to names

        with self.conn.cursor() as cursor:
            # Use the IN clause to query multiple filepaths
            query = """
                SELECT s.sdfloc, n.name 
                FROM structures s 
                JOIN names n ON s.smile = n.smile 
                WHERE s.sdfloc IN %s
            """
            cursor.execute(query, (tuple(str(file) for file in filepaths),))
            results = cursor.fetchall()

        for sdfloc, name in results:
            if file_to_names[Path(sdfloc)] is None:
                file_to_names[Path(sdfloc)] = []
            file_to_names[Path(sdfloc)].append(name)

        failed_idxs = [i for i, file in enumerate(filepaths) if file_to_names[Path(file)] is None]  # Get the indices of failed filepaths
        names = [names for file, names in file_to_names.items() if names is not None]  # Remove None entries

        return names, failed_idxs
    

def get_pharmacophore_data(conformer_files, ph_type_idx, tmp_path: Path = None):

    # create a temporary directory if one is not provided
    delete_tmp_dir = False
    if tmp_path is None:
        delete_tmp_dir = True
        tmp_dir = TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)


    # collect all pharmacophore data
    all_x_pharm = []
    all_a_pharm = []
    failed_pharm_idxs = []

    for idx, conf_file in enumerate(conformer_files):
        x_pharm, a_pharm = get_lig_only_pharmacophore(conf_file, tmp_path, ph_type_idx)
        if x_pharm is None:
            failed_pharm_idxs.append(idx)
            continue
        all_x_pharm.append(x_pharm)
        all_a_pharm.append(a_pharm)

    # delete temporary directory if it was created
    if delete_tmp_dir:
        tmp_dir.cleanup()

    return all_x_pharm, all_a_pharm, failed_pharm_idxs


def save_chunk_to_disk(tensors, chunk_data_file, chunk_info_file):
    
    positions = tensors['positions']
    atom_types = tensors['atom_types']
    atom_charges = tensors['atom_charges']
    bond_types = tensors['bond_types']
    bond_idxs = tensors['bond_idxs']
    x_pharm = tensors['x_pharm']
    a_pharm = tensors['a_pharm']
    databases = tensors['databases']

    # Record the number of nodes and edges in each molecule and convert to numpy arrays
    batch_num_nodes = np.array([x.shape[0] for x in positions])
    batch_num_edges = np.array([eidxs.shape[0] for eidxs in bond_idxs])
    batch_num_pharm_nodes = np.array([x.shape[0] for x in x_pharm])

    # concatenate all the data together
    x = np.concatenate(positions, axis=0)
    a = np.concatenate(atom_types, axis=0)
    c = np.concatenate(atom_charges, axis=0)
    e = np.concatenate(bond_types, axis=0)
    edge_index = np.concatenate(bond_idxs, axis=0)
    x_pharm = np.concatenate(x_pharm, axis=0)
    a_pharm = np.concatenate(a_pharm, axis=0)
    db = np.concatenate(databases, axis=0)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
    node_lookup = build_lookup_table(batch_num_nodes)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
    edge_lookup = build_lookup_table(batch_num_edges)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's pharmacophore node features
    pharm_node_lookup = build_lookup_table(batch_num_pharm_nodes)

    # Tensor dictionary
    chunk_data_dict = { 
        'lig_x': x,
        'lig_a': a,
        'lig_c': c,
        'node_lookup': node_lookup,
        'lig_e': e,
        'lig_edge_idx': edge_index,
        'edge_lookup': edge_lookup,
        'pharm_x': x_pharm,
        'pharm_a': a_pharm,
        'pharm_lookup': pharm_node_lookup,
        'database': databases
    }


    # Save tensor dictionary to npz file
    with open(chunk_data_file, 'wb') as f:
        np.savez(f, **chunk_data_dict)
    

    
    # Chunk data file info dictionary
    chunk_info_dict = {
        'File': chunk_data_file,
        'Mols': len(node_lookup),
        'Atoms': len(x),
        'Edges': len(e),
        'Pharm': len(x_pharm)
    }

    
    # Dump info dictionary in pickle files
    with open(chunk_info_file, "wb") as f:
        pickle.dump(chunk_info_dict, f)

    print('Wrote chunk:', chunk_info_dict['File'])


def generate_library_tensor(names, database_list):
    """
    Generates a binary tensor indicating whether each molecule belongs to any of the specified libraries.

    Args:
        names (list of list of str): A list of lists containing the database names for each molecule.

    Returns:
        np.ndarray: A binary tensor of shape (num_mols, num_libraries) where each element is 1 if the molecule belongs to the library, otherwise 0.
    """
    num_mols = len(names)
    num_libraries = len(database_list)
    
    # Initialize the binary tensor with zeros
    library_tensor = np.zeros((num_mols, num_libraries), dtype=int)
    
    for i, molecule_names in enumerate(names):
        for j, db in enumerate(database_list):
            if db in molecule_names:
                library_tensor[i, j] = 1
    
    return library_tensor
    

def compute_rmsd(mol1, mol2):
    return Chem.CalcRMS(mol1, mol2)


def minimize_molecule(molecule: Chem.rdchem.Mol):

    # create a copy of the original ligand
    lig = Chem.Mol(molecule)

    # Add hydrogens
    lig_H = Chem.AddHs(lig, addCoords=True)
    Chem.SanitizeMol(lig_H)

    try:
        ff = Chem.UFFGetMoleculeForceField(lig_H,ignoreInterfragInteractions=False)
    except Exception as e:
        print("Failed to get force field:", e)
        return None

    # documentation for this function call, incase we want to play with number of minimization steps or record whether it was successful: https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html#rdkit.ForceField.rdForceField.ForceField.Minimize
    try:
        ff.Minimize(maxIts=400)
    except Exception as e:
        print("Failed to minimize molecule")
        return None

    # Get the minimized positions for molecule with H's
    cpos = lig_H.GetConformer().GetPositions()

    # Original ligand with no H's
    conf = lig.GetConformer()

    for (i,xyz) in enumerate(cpos[-lig.GetNumAtoms():]):
        conf.SetAtomPosition(i,xyz)
    
    return lig


def remove_counterions_batch(mols: list[Chem.Mol], counterions: list[str]):
    for idx in range(len(mols)):
        mol = mols[idx]
        for i, atom in enumerate(mol.GetAtoms()):
            if str(atom.GetSymbol()) in counterions:
                print(f"Atom {atom.GetSymbol()} is a known counterion. Removing and minimizing structure.")
                mol_cpy = Chem.EditableMol(mol)
                mol_cpy.RemoveAtom(i)
                mol_cpy = mol_cpy.GetMol()
                mol = minimize_molecule(mol_cpy)
                mols[idx] = mol
    return mols
    