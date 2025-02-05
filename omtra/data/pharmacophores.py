import numpy as np
import rdkit.Chem as Chem
from scipy.spatial.distance import cdist
from pharmvec import GetDonorFeatVects, GetAcceptorFeatVects, GetAromaticFeatVects
import pprint

smarts_patterns = {
    'Aromatic': ["a1aaaaa1", "a1aaaa1"],
    'PositiveIon': ['[+,+2,+3,+4]', "[$(C(N)(N)=N)]", "[$(n1cc[nH]c1)]"],
    'NegativeIon': ['[-,-2,-3,-4]', "C(=O)[O-,OH,OX1]"],
    'HydrogenAcceptor': [
        "[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]", 
        "[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]"
    ],
    'HydrogenDonor': [
        "[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]", "[#8!H0&!$([OH][C,S,P]=O)]", "[#16!H0]"
    ],
    'Hydrophobic': [
        "a1aaaaa1", "a1aaaa1", 
        "[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
        "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
        "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
        "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
        "[$([S]~[#6])&!$(S~[!#6])]"
    ]
}

def get_smarts_matches(rdmol, smarts_pattern):
    """Find positions of a SMARTS pattern in molecule."""
    feature_positions = []
    atom_positions = []
    smarts_mol = Chem.MolFromSmarts(smarts_pattern)
    matches = rdmol.GetSubstructMatches(smarts_mol, uniquify=True)
    for match in matches:
        atoms = [np.array(rdmol.GetConformer().GetAtomPosition(idx)) for idx in match]
        feature_location = np.mean(atoms, axis=0)
        
        atom_positions.append(atoms)
        feature_positions.append(feature_location)

    return matches, atom_positions, feature_positions

# get the vectors for all matches of a smarts pattern
def get_vectors(mol, feature, atom_idxs, atom_positions, feature_positions):
    vectors = []
    for featAtoms, atomsLoc, featLoc in zip(atom_idxs, atom_positions, feature_positions):
        if feature == 'Aromatic':
            vectors.append(GetAromaticFeatVects(atomsLoc, featLoc, scale=1.0)[0]) #pick one of the vectors
        elif feature == 'HydrogenDonor':
            vectors.append(GetDonorFeatVects(featAtoms, atomsLoc, mol))
        elif feature == 'HydrogenAcceptor':
            vectors.append(GetAcceptorFeatVects(featAtoms, atomsLoc, mol))
        else:
            vectors.append(np.zeros(3))
    return vectors


def get_pharmacophore_dict(mol):
    """Extract pharmacophores and direction vectors from RDKit molecule.
        
    Returns
    -------
    dictionary : {'FeatureName' : {
                                   'P': [(coord), ... ],
                                   'V': [(vec), ... ]
                                   }
                 }
    """
    
    pharmacophores = {feature: {'P': [], 'V': []} for feature in smarts_patterns}

    for feature, patterns in smarts_patterns.items():
        for pattern in patterns:
            atom_idxs, atom_positions, feature_positions = get_smarts_matches(mol, pattern)
            if feature_positions:
                vectors = get_vectors(mol, feature, atom_idxs, atom_positions, feature_positions)
                pharmacophores[feature]['P'].extend(feature_positions)
                pharmacophores[feature]['V'].extend(vectors)

    return pharmacophores

'''
supplier = Chem.SDMolSupplier('test.sdf')
mol = supplier[0]
dict = get_pharmacophore_dict(mol)
pprint.pprint(dict)
'''