"""This module contains access to empirical distributions from the plinder dataset."""

from omtra.utils import omtra_root

import numpy as np
import pandas as pd
from pathlib import Path

def plinder_n_nodes_dist() -> dict:
    plinder_dists_file = Path(omtra_root()) / "omtra_pipelines/plinder_dataset/plinder_filtered.parquet"

    # read dataframe
    df = pd.read_parquet(plinder_dists_file)

    # filter out npndes
    df = df[ df['ligand_type'] == 'ligand' ]

    # get the observed counts of (n_ligand_atoms, n_pharmacophores, n_protein_atoms)
    raw_cols = ['num_heavy_atoms', 'num_pharmacophores', 'num_pocket_atoms']
    observed = df[raw_cols].values.astype(int)
    lpp_unique, lpp_counts = np.unique(observed, axis=0, return_counts=True)

    # get the support (the set of unique values observed for dimension in the joint distribution)
    supports = {
        'n_ligand_atoms': np.sort(np.unique(lpp_unique[:, 0])),
        'n_pharms': np.sort(np.unique(lpp_unique[:, 1])),
        'n_protein_atoms': np.sort(np.unique(lpp_unique[:, 2]))
    }

    # convert counts to the full joint distribution p(n_ligand_atoms, n_pharmacophores, n_protein_atoms)
    p_lpp = np.zeros((len(supports['n_ligand_atoms']), len(supports['n_pharms']), len(supports['n_protein_atoms'])), dtype=float)
    for lpp_observed, lpp_count in zip(lpp_unique, lpp_counts):
        lig_idx = np.where(supports['n_ligand_atoms'] == lpp_observed[0])[0][0]
        pharm_idx = np.where(supports['n_pharms'] == lpp_observed[1])[0][0]
        prot_idx = np.where(supports['n_protein_atoms'] == lpp_observed[2])[0][0]
        p_lpp[lig_idx, pharm_idx, prot_idx] = lpp_count

    p_lpp = p_lpp / np.sum(p_lpp)

    output = {
        'density': p_lpp,
        'supports': supports,
    }
    return output