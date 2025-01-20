import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import MySQLdb
import itertools

# TODO: this script should actually take as input just a hydra config 
# - but Ramith is setting up our hydra stuff yet, and we don't 
# yet know what the config for this dataset processing component will look like
# so for now just argparse, and once its written it'll be easy/concrete to 
# port into a hydra config
def parse_args():
    p = argparse.ArgumentParser(description='Process pharmit data')

    # temporary default path for development
    p.add_argument('--conf_file', type=Path, default='../tmp_conformer_inspection/100.sdf.gz')
    p.add_argument('--skip_query', action='store_true', help='Skip querying the database for names')

    args = p.parse_args()
    return args

def extract_pharmacophore_data(mol):
    """
    Parses pharmacophore data from an RDKit molecule object into a dictionary.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object containing pharmacophore data.

    Returns:
        dict: Parsed pharmacophore data with types as keys and lists of tuples as values.
    """
    pharmacophore_data = mol.GetProp("pharmacophore") if mol.HasProp("pharmacophore") else None
    if pharmacophore_data is None:
        return None

    parsed_data = []
    lines = pharmacophore_data.splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            ph_type = parts[0]  # Pharmacophore type
            try:
                coordinates = tuple(map(float, parts[1:4]))  # Extract the 3 float values
                parsed_data.append((ph_type, coordinates))
            except ValueError:
                print(f"Skipping line due to parsing error: {line}")

    return parsed_data

class NameFinder():

    def __init__(self):
        self.conn = MySQLdb.connect(host="localhost", user="pharmit", db="conformers")
        self.cursor = self.conn.cursor()

    def query(self, smiles: str):
        self.cursor.execute("SELECT name FROM names WHERE smile = %s", (smiles,))
        names = self.cursor.fetchall()
        names = list(itertools.chain.from_iterable(names)) 
        return names

if __name__ == '__main__':
    args = parse_args()


    # get the first conformer from the file
    conformer_file_path = args.conf_file
    with gzip.open(conformer_file_path, 'rb') as gzipped_sdf:
        suppl = Chem.ForwardSDMolSupplier(gzipped_sdf)
        for mol in suppl:
            if mol is not None:
                break
        if mol is None:
            raise ValueError(f"Failed to parse a molecule from {conformer_file_path}")
        
    # extract pharmacophore data from the molecule
    pharmacophore_data = extract_pharmacophore_data(mol)

    # get smiles string
    smiles = Chem.MolToSmiles(mol)

    print(f"Pharmacophore data: {pharmacophore_data}")
    print(f"SMILES: {smiles}")
    
    if not args.skip_query:
        name_finder = NameFinder()
        names = name_finder.query(smiles)
        print(f"Names: {names}")
        
