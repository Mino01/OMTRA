import torch
import functools
from omtra.utils.misc import classproperty
from copy import deepcopy

import omtra.tasks.prior_collections as pc

from omtra.tasks.register import register_task

canonical_modality_order = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']
canonical_entity_order = ['ligand', 'protein', 'pharmacophore']

class Task:
    protein_state_t0 = 'noise'

    @classproperty
    def t0_modality_arr(self) -> torch.Tensor:
        arr = torch.zeros(len(canonical_modality_order), dtype=bool)
        for i, modality in enumerate(canonical_modality_order):
            if modality in self.observed_at_t0:
                arr[i] = 1
        return arr
    
    @classproperty
    def t1_modality_arr(cls) -> torch.Tensor:
        arr = torch.zeros(len(canonical_modality_order), dtype=bool)
        for i, modality in enumerate(canonical_modality_order):
            if modality in cls.observed_at_t1:
                arr[i] = 1
        return arr
    
    @classproperty
    def modalities_present(self):
        present_modality_mask = self.t0_modality_arr | self.t1_modality_arr
        present_modality_idxs = torch.where(present_modality_mask)[0]
        return [canonical_modality_order[i] for i in present_modality_idxs]
    
    @classproperty
    def uses_apo(self):
        # TODO: this logic is subject to change if we ever decide to do apo sampling as a task
        # because then there would be a task where the intial protein state is noise but we still require the apo state (it would be the target)
        # but i think sampling apo states is not a very useful task for the application of sbdd
        return self.protein_state_t0 == 'apo'
    
##
# tasks with ligand only
##
@register_task("denovo_ligand")
class DeNovoLigand(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure']

    priors = dict(lig=pc.denovo_ligand)

@register_task("ligand_conformer")
class LigandConformer(Task):
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure']

    priors = {
        'lig': pc.ligand_conformer
    }

## 
# tasks with ligand + pharmacophore
##
@register_task("denovo_ligand_pharmacophore")
class DeNovoLigandPharmacophore(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']

    priors = {
        'lig': pc.denovo_ligand,
        'pharm': pc.denovo_pharmacophore
    }


# while technically possible, i don't think this task is very useful?
# @register_task("ligand_conformer_pharmacophore")
# class LigandConformerPharmacophore(Task):
#     observed_at_t0 = ['ligand_identity']
#     observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']

##
# tasks with ligand+protein and no pharmacophore
##
@register_task("protein_ligand_denovo")
class ProteinLigandDeNovo(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']

    priors = {
        'lig': pc.denovo_ligand
    }
    priors['protein'] = {
        'type': 'dd_gaussian',
        'params': {'std': 2.0}
    }


@register_task("apo_conditioned_denovo_ligand")
class ApoConditionedDeNovoLigand(Task):
    observed_at_t0 = ['protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure']
    protein_state_t0 = 'apo'

@register_task("unconditional_ligand_docking")
class UnconditionalLigandDocking(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']

@register_task("apo_conditioned_ligand_docking")
class ApoConditionedLigandDocking(Task):
    """Docking a ligand into the protein structure, assuming knowledge of the apo protein structure at t=0"""
    observed_at_t0 = ['ligand_identity', 'protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']
    protein_state_t0 = 'apo'

##
# Tasks with ligand+protein+pharmacophore
##
@register_task("protein_ligand_pharmacophore_denovo")
class ProteinLigandPharmacophoreDeNovo(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']

@register_task("apo_conditioned_denovo_ligand_pharmacophore")
class ApoConditionedDeNovoLigandPharmacophore(Task):
    observed_at_t0 = ['protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']
    protein_state_t0 = 'apo'

@register_task("unconditional_ligand_docking_pharmacophore")
class UnconditionalLigandDockingPharmacophore(Task):
    """Docking a ligand into the protein while generating a pharmacophore, assuming no knowledge of the protein structure at t=0"""
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']

@register_task("apo_conditioned_ligand_docking_pharmacophore")
class ApoConditionedLigandDockingPharmacophore(Task):
    """Docking a ligand into the protein while generating a pharmacophore, assuming knowledge of the apo protein structure at t=0"""
    observed_at_t0 = ['ligand_identity', 'protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']
    protein_state_t0 = 'apo'

## 
# Tasks with protein+pharmacophore and no ligand
##
@register_task("protein_pharmacophore")
class ProteinPharmacophore(Task):
    observed_at_t0 = []
    observed_at_t1 = ['protein', 'pharmacophore']

@register_task("apo_conditioned_protein_pharmacophore")
class ApoConditionedProteinPharmacophore(Task):
    observed_at_t0 = ['protein']
    observed_at_t1 = ['protein', 'pharmacophore']
    protein_state_t0 = 'apo'

##
# Tasks with protein only
## 
@register_task("apo_protein_sampling")
class ApoProteinSampling(Task):
    """Sampling apo protein conformations, starting from noise for the protein at t=0"""
    observed_at_t0 = []
    observed_at_t1 = ['protein']

@register_task("apo_to_holo_protein")
class ApotoHoloProtein(Task):
    """Predicting the holo protein structure, starting from the apo protein structure at t=0"""
    observed_at_t0 = ['protein']
    observed_at_t1 = ['protein']
    protein_state_t0 = 'apo'


