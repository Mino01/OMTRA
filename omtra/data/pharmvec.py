import numpy as np

def GetAromaticFeatVects(atomsLoc, featLoc, return_both: bool = False, scale: float = 1.0):
    """Compute the direction vector for an aromatic feature."""
    
    v1 = atomsLoc[0] - featLoc
    v2 = atomsLoc[1] - featLoc

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    normal *= scale
    
    if return_both:
        normal2 = normal * -1
        return [normal, normal2]
    else:
        return [normal]


def GetDonorFeatVects(featAtoms, atomsLoc, rdmol):
    atom_idx = featAtoms[0]
    atom_coords = atomsLoc[0]
    vectors = []
    
    for nbor in rdmol.GetAtomWithIdx(atom_idx).GetNeighbors():
        print(nbor.GetAtomicNum())
        if nbor.GetAtomicNum() == 1:  # hydrogen atom
            nbor_coords = np.array(rdmol.GetConformer().GetAtomPosition(nbor.GetIdx()))
            vec = nbor_coords - atom_coords
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        
    return vectors


def GetAcceptorFeatVects(featAtoms, atomsLoc, rdmol):
    atom_idx = featAtoms[0]
    atom_coords = atomsLoc[0]
    vectors = []
    found_vec = False

    # check if any hydrogen neighbor exists
    for nbor in rdmol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if nbor.GetAtomicNum() == 1:
            nbor_coords = np.array(rdmol.GetConformer().GetAtomPosition(nbor.GetIdx()))
            vec = nbor_coords - atom_coords
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            found_vec = True

    if not found_vec:
        # compute average direction of bonds and reverse it
        ave_bond = np.zeros(3)
        cnt = 0
        
        for nbor in rdmol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nbor_coords = np.array(rdmol.GetConformer().GetAtomPosition(nbor.GetIdx()))
            ave_bond += nbor_coords - atom_coords 
            cnt += 1
        
        if cnt > 0:
            ave_bond /= cnt
            ave_bond *= -1  # reverse the direction
            ave_bond = ave_bond / np.linalg.norm(ave_bond)
            vectors.append(ave_bond)

    return vectors
