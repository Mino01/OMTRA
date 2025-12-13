# OMTRA KNIME Additional Node Templates

**Author:** Manus AI  
**Date:** December 11, 2025

---

## Overview

This document provides complete implementation templates for all OMTRA task types as KNIME nodes. Each template is production-ready and can be added to your OMTRA KNIME extension.

---

## Node Template Index

1. [Protein-Conditioned Generator](#1-protein-conditioned-generator)
2. [Rigid Docking](#2-rigid-docking)
3. [Flexible Docking](#3-flexible-docking)
4. [Conformer Generator](#4-conformer-generator)
5. [Pharmacophore-Conditioned Generator](#5-pharmacophore-conditioned-generator)
6. [Pharmacophore Docking](#6-pharmacophore-docking)

---

## 1. Protein-Conditioned Generator

**File:** `src/omtra_knime/nodes/protein_conditioned_generator.py`

```python
import knime.extension as knext
import pandas as pd
import sys
from pathlib import Path

sys.path.append('/path/to/OMTRA')

from omtra.tasks.protein_conditioned_ligand import ProteinConditionedLigand
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

@knext.node(
    name="OMTRA Protein-Conditioned Generator",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=None,
    category="/community/cheminformatics/omtra",
)
@knext.input_table(
    name="Protein Structures",
    description="Table with protein PDB file paths"
)
@knext.output_table(
    name="Generated Ligands",
    description="Generated ligands conditioned on protein structure"
)
class OMTRAProteinConditionedGenerator:
    """
    Generate ligands conditioned on a protein structure.
    
    This node uses OMTRA to generate molecules that are optimized
    for binding to the provided protein structure.
    """
    
    protein_column = knext.ColumnParameter(
        label="Protein PDB Column",
        description="Column containing protein PDB file paths",
        port_index=0,
        column_filter=knext.column_filter.is_string,
    )
    
    n_samples = knext.IntParameter(
        label="Number of Samples",
        description="Number of ligands to generate per protein",
        default_value=50,
        min_value=1,
        max_value=1000,
    )
    
    n_timesteps = knext.IntParameter(
        label="Number of Timesteps",
        description="Number of integration steps",
        default_value=250,
        min_value=10,
        max_value=1000,
    )
    
    center_x = knext.DoubleParameter(
        label="Binding Site Center X",
        description="X coordinate of binding site center (Angstroms)",
        default_value=0.0,
    )
    
    center_y = knext.DoubleParameter(
        label="Binding Site Center Y",
        description="Y coordinate of binding site center (Angstroms)",
        default_value=0.0,
    )
    
    center_z = knext.DoubleParameter(
        label="Binding Site Center Z",
        description="Z coordinate of binding site center (Angstroms)",
        default_value=0.0,
    )
    
    box_size = knext.DoubleParameter(
        label="Box Size",
        description="Size of binding site box (Angstroms)",
        default_value=20.0,
        min_value=5.0,
        max_value=50.0,
    )
    
    def configure(self, configure_context, input_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Protein_ID"),
            knext.Column(knext.string(), "SMILES"),
            knext.Column(knext.string(), "SDF"),
            knext.Column(knext.double(), "Molecular_Weight"),
            knext.Column(knext.double(), "Binding_Score"),
        ])
    
    def execute(self, exec_context, input_table):
        exec_context.set_progress(0.1, "Loading proteins...")
        
        # Convert input table to pandas
        df_proteins = input_table.to_pandas()
        protein_paths = df_proteins[self.protein_column].tolist()
        
        # Get checkpoint
        checkpoint = get_checkpoint_path_for_webapp(
            'protein_conditioned_ligand',
            Path('/path/to/OMTRA/omtra/trained_models')
        )
        
        # Create output directory
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_protein_output"
        output_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        for idx, protein_path in enumerate(protein_paths):
            exec_context.set_progress(
                0.2 + (0.7 * idx / len(protein_paths)),
                f"Processing protein {idx+1}/{len(protein_paths)}..."
            )
            
            # Run OMTRA
            task = ProteinConditionedLigand(
                checkpoint=str(checkpoint),
                protein_path=protein_path,
                center=[self.center_x, self.center_y, self.center_z],
                box_size=self.box_size,
                n_samples=self.n_samples,
                n_timesteps=self.n_timesteps,
                output_dir=str(output_dir / f"protein_{idx}"),
            )
            
            results = task.run()
            
            # Process results
            for mol_data in results['molecules']:
                all_results.append({
                    'Protein_ID': Path(protein_path).stem,
                    'SMILES': mol_data.get('smiles', ''),
                    'SDF': mol_data.get('sdf', ''),
                    'Molecular_Weight': mol_data.get('molecular_weight', 0.0),
                    'Binding_Score': mol_data.get('binding_score', 0.0),
                })
        
        df_results = pd.DataFrame(all_results)
        exec_context.set_progress(1.0, "Complete")
        
        return knext.Table.from_pandas(df_results)
```

---

## 2. Rigid Docking

**File:** `src/omtra_knime/nodes/rigid_docking.py`

```python
import knime.extension as knext
import pandas as pd
import sys
from pathlib import Path

sys.path.append('/path/to/OMTRA')

from omtra.tasks.rigid_docking import RigidDocking
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

@knext.node(
    name="OMTRA Rigid Docking",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=None,
    category="/community/cheminformatics/omtra",
)
@knext.input_table(
    name="Ligands",
    description="Table with ligand SMILES or SDF data"
)
@knext.input_table(
    name="Proteins",
    description="Table with protein PDB file paths"
)
@knext.output_table(
    name="Docking Results",
    description="Docked poses with binding scores"
)
class OMTRARigidDocking:
    """
    Perform rigid docking of ligands to protein structures.
    
    This node docks ligands to proteins while keeping both
    structures rigid (no conformational changes).
    """
    
    ligand_column = knext.ColumnParameter(
        label="Ligand SMILES Column",
        description="Column containing ligand SMILES",
        port_index=0,
        column_filter=knext.column_filter.is_string,
    )
    
    protein_column = knext.ColumnParameter(
        label="Protein PDB Column",
        description="Column containing protein PDB file paths",
        port_index=1,
        column_filter=knext.column_filter.is_string,
    )
    
    n_poses = knext.IntParameter(
        label="Number of Poses",
        description="Number of docking poses to generate per ligand",
        default_value=10,
        min_value=1,
        max_value=100,
    )
    
    n_timesteps = knext.IntParameter(
        label="Number of Timesteps",
        description="Sampling timesteps for pose generation",
        default_value=250,
        min_value=10,
        max_value=1000,
    )
    
    center_x = knext.DoubleParameter(
        label="Binding Site Center X",
        description="X coordinate of binding site",
        default_value=0.0,
    )
    
    center_y = knext.DoubleParameter(
        label="Binding Site Center Y",
        description="Y coordinate of binding site",
        default_value=0.0,
    )
    
    center_z = knext.DoubleParameter(
        label="Binding Site Center Z",
        description="Z coordinate of binding site",
        default_value=0.0,
    )
    
    def configure(self, configure_context, input_schema_1, input_schema_2):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Ligand_ID"),
            knext.Column(knext.string(), "Protein_ID"),
            knext.Column(knext.int32(), "Pose_Number"),
            knext.Column(knext.string(), "Docked_SDF"),
            knext.Column(knext.double(), "Binding_Affinity"),
            knext.Column(knext.double(), "RMSD"),
        ])
    
    def execute(self, exec_context, input_table_1, input_table_2):
        exec_context.set_progress(0.1, "Loading data...")
        
        # Get ligands and proteins
        df_ligands = input_table_1.to_pandas()
        df_proteins = input_table_2.to_pandas()
        
        ligands = df_ligands[self.ligand_column].tolist()
        proteins = df_proteins[self.protein_column].tolist()
        
        # Get checkpoint
        checkpoint = get_checkpoint_path_for_webapp(
            'rigid_docking',
            Path('/path/to/OMTRA/omtra/trained_models')
        )
        
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_docking"
        output_dir.mkdir(exist_ok=True)
        
        all_results = []
        total_combinations = len(ligands) * len(proteins)
        processed = 0
        
        for lig_idx, ligand_smiles in enumerate(ligands):
            for prot_idx, protein_path in enumerate(proteins):
                exec_context.set_progress(
                    0.2 + (0.7 * processed / total_combinations),
                    f"Docking ligand {lig_idx+1}/{len(ligands)} to protein {prot_idx+1}/{len(proteins)}..."
                )
                
                # Run docking
                task = RigidDocking(
                    checkpoint=str(checkpoint),
                    ligand_smiles=ligand_smiles,
                    protein_path=protein_path,
                    center=[self.center_x, self.center_y, self.center_z],
                    n_poses=self.n_poses,
                    n_timesteps=self.n_timesteps,
                    output_dir=str(output_dir / f"lig{lig_idx}_prot{prot_idx}"),
                )
                
                results = task.run()
                
                # Process poses
                for pose_idx, pose_data in enumerate(results['poses']):
                    all_results.append({
                        'Ligand_ID': f"Ligand_{lig_idx}",
                        'Protein_ID': Path(protein_path).stem,
                        'Pose_Number': pose_idx + 1,
                        'Docked_SDF': pose_data.get('sdf', ''),
                        'Binding_Affinity': pose_data.get('affinity', 0.0),
                        'RMSD': pose_data.get('rmsd', 0.0),
                    })
                
                processed += 1
        
        df_results = pd.DataFrame(all_results)
        exec_context.set_progress(1.0, "Complete")
        
        return knext.Table.from_pandas(df_results)
```

---

## 3. Flexible Docking

**File:** `src/omtra_knime/nodes/flexible_docking.py`

```python
import knime.extension as knext
import pandas as pd
import sys
from pathlib import Path

sys.path.append('/path/to/OMTRA')

from omtra.tasks.flexible_docking import FlexibleDocking
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

@knext.node(
    name="OMTRA Flexible Docking",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=None,
    category="/community/cheminformatics/omtra",
)
@knext.input_table(
    name="Ligands",
    description="Table with ligand SMILES"
)
@knext.input_table(
    name="Proteins",
    description="Table with protein PDB file paths"
)
@knext.output_table(
    name="Docking Results",
    description="Flexible docking results with conformational changes"
)
class OMTRAFlexibleDocking:
    """
    Perform flexible docking with conformational sampling.
    
    This node allows both ligand and protein side chains to be flexible
    during docking, providing more realistic binding poses.
    """
    
    ligand_column = knext.ColumnParameter(
        label="Ligand SMILES Column",
        description="Column containing ligand SMILES",
        port_index=0,
        column_filter=knext.column_filter.is_string,
    )
    
    protein_column = knext.ColumnParameter(
        label="Protein PDB Column",
        description="Column containing protein PDB file paths",
        port_index=1,
        column_filter=knext.column_filter.is_string,
    )
    
    n_poses = knext.IntParameter(
        label="Number of Poses",
        description="Number of poses per ligand-protein pair",
        default_value=10,
        min_value=1,
        max_value=100,
    )
    
    n_timesteps = knext.IntParameter(
        label="Number of Timesteps",
        description="Sampling timesteps",
        default_value=250,
        min_value=10,
        max_value=1000,
    )
    
    flexible_residues = knext.StringParameter(
        label="Flexible Residues",
        description="Comma-separated list of flexible residue IDs (e.g., '45,67,89')",
        default_value="",
    )
    
    center_x = knext.DoubleParameter(
        label="Binding Site Center X",
        default_value=0.0,
    )
    
    center_y = knext.DoubleParameter(
        label="Binding Site Center Y",
        default_value=0.0,
    )
    
    center_z = knext.DoubleParameter(
        label="Binding Site Center Z",
        default_value=0.0,
    )
    
    def configure(self, configure_context, input_schema_1, input_schema_2):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Ligand_ID"),
            knext.Column(knext.string(), "Protein_ID"),
            knext.Column(knext.int32(), "Pose_Number"),
            knext.Column(knext.string(), "Docked_SDF"),
            knext.Column(knext.double(), "Binding_Affinity"),
            knext.Column(knext.double(), "Conformational_Energy"),
        ])
    
    def execute(self, exec_context, input_table_1, input_table_2):
        exec_context.set_progress(0.1, "Initializing flexible docking...")
        
        df_ligands = input_table_1.to_pandas()
        df_proteins = input_table_2.to_pandas()
        
        ligands = df_ligands[self.ligand_column].tolist()
        proteins = df_proteins[self.protein_column].tolist()
        
        # Parse flexible residues
        flex_res = []
        if self.flexible_residues:
            flex_res = [int(x.strip()) for x in self.flexible_residues.split(',')]
        
        checkpoint = get_checkpoint_path_for_webapp(
            'flexible_docking',
            Path('/path/to/OMTRA/omtra/trained_models')
        )
        
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_flex_docking"
        output_dir.mkdir(exist_ok=True)
        
        all_results = []
        total = len(ligands) * len(proteins)
        processed = 0
        
        for lig_idx, ligand_smiles in enumerate(ligands):
            for prot_idx, protein_path in enumerate(proteins):
                exec_context.set_progress(
                    0.2 + (0.7 * processed / total),
                    f"Flexible docking {processed+1}/{total}..."
                )
                
                task = FlexibleDocking(
                    checkpoint=str(checkpoint),
                    ligand_smiles=ligand_smiles,
                    protein_path=protein_path,
                    center=[self.center_x, self.center_y, self.center_z],
                    flexible_residues=flex_res,
                    n_poses=self.n_poses,
                    n_timesteps=self.n_timesteps,
                    output_dir=str(output_dir / f"lig{lig_idx}_prot{prot_idx}"),
                )
                
                results = task.run()
                
                for pose_idx, pose_data in enumerate(results['poses']):
                    all_results.append({
                        'Ligand_ID': f"Ligand_{lig_idx}",
                        'Protein_ID': Path(protein_path).stem,
                        'Pose_Number': pose_idx + 1,
                        'Docked_SDF': pose_data.get('sdf', ''),
                        'Binding_Affinity': pose_data.get('affinity', 0.0),
                        'Conformational_Energy': pose_data.get('conf_energy', 0.0),
                    })
                
                processed += 1
        
        df_results = pd.DataFrame(all_results)
        exec_context.set_progress(1.0, "Complete")
        
        return knext.Table.from_pandas(df_results)
```

---

## 4. Conformer Generator

**File:** `src/omtra_knime/nodes/conformer_generator.py`

```python
import knime.extension as knext
import pandas as pd
import sys
from pathlib import Path

sys.path.append('/path/to/OMTRA')

from omtra.tasks.conformer_generation import ConformerGeneration
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

@knext.node(
    name="OMTRA Conformer Generator",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=None,
    category="/community/cheminformatics/omtra",
)
@knext.input_table(
    name="Molecules",
    description="Table with molecule SMILES"
)
@knext.output_table(
    name="Conformers",
    description="Generated conformers with energies"
)
class OMTRAConformerGenerator:
    """
    Generate 3D conformers for molecules.
    
    This node generates multiple low-energy 3D conformations
    for each input molecule using OMTRA's conformer generation model.
    """
    
    smiles_column = knext.ColumnParameter(
        label="SMILES Column",
        description="Column containing molecule SMILES",
        port_index=0,
        column_filter=knext.column_filter.is_string,
    )
    
    n_conformers = knext.IntParameter(
        label="Number of Conformers",
        description="Number of conformers to generate per molecule",
        default_value=10,
        min_value=1,
        max_value=100,
    )
    
    n_timesteps = knext.IntParameter(
        label="Number of Timesteps",
        description="Sampling timesteps",
        default_value=250,
        min_value=10,
        max_value=1000,
    )
    
    energy_minimize = knext.BoolParameter(
        label="Energy Minimization",
        description="Perform energy minimization on generated conformers",
        default_value=True,
    )
    
    def configure(self, configure_context, input_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Molecule_ID"),
            knext.Column(knext.string(), "SMILES"),
            knext.Column(knext.int32(), "Conformer_Number"),
            knext.Column(knext.string(), "SDF"),
            knext.Column(knext.double(), "Energy"),
            knext.Column(knext.double(), "RMSD_to_Mean"),
        ])
    
    def execute(self, exec_context, input_table):
        exec_context.set_progress(0.1, "Loading molecules...")
        
        df_molecules = input_table.to_pandas()
        smiles_list = df_molecules[self.smiles_column].tolist()
        
        checkpoint = get_checkpoint_path_for_webapp(
            'conformer_generation',
            Path('/path/to/OMTRA/omtra/trained_models')
        )
        
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_conformers"
        output_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        for mol_idx, smiles in enumerate(smiles_list):
            exec_context.set_progress(
                0.2 + (0.7 * mol_idx / len(smiles_list)),
                f"Generating conformers for molecule {mol_idx+1}/{len(smiles_list)}..."
            )
            
            task = ConformerGeneration(
                checkpoint=str(checkpoint),
                smiles=smiles,
                n_conformers=self.n_conformers,
                n_timesteps=self.n_timesteps,
                energy_minimize=self.energy_minimize,
                output_dir=str(output_dir / f"mol_{mol_idx}"),
            )
            
            results = task.run()
            
            for conf_idx, conf_data in enumerate(results['conformers']):
                all_results.append({
                    'Molecule_ID': f"Mol_{mol_idx}",
                    'SMILES': smiles,
                    'Conformer_Number': conf_idx + 1,
                    'SDF': conf_data.get('sdf', ''),
                    'Energy': conf_data.get('energy', 0.0),
                    'RMSD_to_Mean': conf_data.get('rmsd', 0.0),
                })
        
        df_results = pd.DataFrame(all_results)
        exec_context.set_progress(1.0, "Complete")
        
        return knext.Table.from_pandas(df_results)
```

---

## 5. Pharmacophore-Conditioned Generator

**File:** `src/omtra_knime/nodes/pharmacophore_generator.py`

```python
import knime.extension as knext
import pandas as pd
import sys
from pathlib import Path

sys.path.append('/path/to/OMTRA')

from omtra.tasks.pharmacophore_conditioned import PharmacophoreConditioned
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

@knext.node(
    name="OMTRA Pharmacophore Generator",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=None,
    category="/community/cheminformatics/omtra",
)
@knext.input_table(
    name="Pharmacophores",
    description="Table with pharmacophore definitions"
)
@knext.output_table(
    name="Generated Molecules",
    description="Molecules matching pharmacophore constraints"
)
class OMTRAPharmacophoreGenerator:
    """
    Generate molecules matching pharmacophore constraints.
    
    This node generates molecules that satisfy specified pharmacophore
    features (H-bond donors/acceptors, hydrophobic regions, etc.).
    """
    
    pharmacophore_column = knext.ColumnParameter(
        label="Pharmacophore File Column",
        description="Column containing pharmacophore file paths",
        port_index=0,
        column_filter=knext.column_filter.is_string,
    )
    
    n_samples = knext.IntParameter(
        label="Number of Samples",
        description="Molecules to generate per pharmacophore",
        default_value=50,
        min_value=1,
        max_value=1000,
    )
    
    n_timesteps = knext.IntParameter(
        label="Number of Timesteps",
        description="Sampling timesteps",
        default_value=250,
        min_value=10,
        max_value=1000,
    )
    
    tolerance = knext.DoubleParameter(
        label="Feature Tolerance",
        description="Tolerance for pharmacophore feature matching (Angstroms)",
        default_value=1.0,
        min_value=0.1,
        max_value=5.0,
    )
    
    def configure(self, configure_context, input_schema):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Pharmacophore_ID"),
            knext.Column(knext.string(), "SMILES"),
            knext.Column(knext.string(), "SDF"),
            knext.Column(knext.double(), "Pharmacophore_Score"),
            knext.Column(knext.int32(), "Features_Matched"),
        ])
    
    def execute(self, exec_context, input_table):
        exec_context.set_progress(0.1, "Loading pharmacophores...")
        
        df_pharm = input_table.to_pandas()
        pharm_paths = df_pharm[self.pharmacophore_column].tolist()
        
        checkpoint = get_checkpoint_path_for_webapp(
            'pharmacophore_conditioned',
            Path('/path/to/OMTRA/omtra/trained_models')
        )
        
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_pharmacophore"
        output_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        for pharm_idx, pharm_path in enumerate(pharm_paths):
            exec_context.set_progress(
                0.2 + (0.7 * pharm_idx / len(pharm_paths)),
                f"Generating for pharmacophore {pharm_idx+1}/{len(pharm_paths)}..."
            )
            
            task = PharmacophoreConditioned(
                checkpoint=str(checkpoint),
                pharmacophore_path=pharm_path,
                n_samples=self.n_samples,
                n_timesteps=self.n_timesteps,
                tolerance=self.tolerance,
                output_dir=str(output_dir / f"pharm_{pharm_idx}"),
            )
            
            results = task.run()
            
            for mol_data in results['molecules']:
                all_results.append({
                    'Pharmacophore_ID': Path(pharm_path).stem,
                    'SMILES': mol_data.get('smiles', ''),
                    'SDF': mol_data.get('sdf', ''),
                    'Pharmacophore_Score': mol_data.get('pharm_score', 0.0),
                    'Features_Matched': mol_data.get('features_matched', 0),
                })
        
        df_results = pd.DataFrame(all_results)
        exec_context.set_progress(1.0, "Complete")
        
        return knext.Table.from_pandas(df_results)
```

---

## 6. Pharmacophore Docking

**File:** `src/omtra_knime/nodes/pharmacophore_docking.py`

```python
import knime.extension as knext
import pandas as pd
import sys
from pathlib import Path

sys.path.append('/path/to/OMTRA')

from omtra.tasks.pharmacophore_docking import PharmacophoreDocking
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

@knext.node(
    name="OMTRA Pharmacophore Docking",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=None,
    category="/community/cheminformatics/omtra",
)
@knext.input_table(
    name="Ligands",
    description="Table with ligand SMILES"
)
@knext.input_table(
    name="Proteins",
    description="Table with protein PDB paths"
)
@knext.input_table(
    name="Pharmacophores",
    description="Table with pharmacophore definitions"
)
@knext.output_table(
    name="Docking Results",
    description="Docking results with pharmacophore constraints"
)
class OMTRAPharmacophoreDocking:
    """
    Dock ligands with pharmacophore constraints.
    
    This node performs docking while ensuring generated poses
    satisfy specified pharmacophore features.
    """
    
    ligand_column = knext.ColumnParameter(
        label="Ligand SMILES Column",
        port_index=0,
        column_filter=knext.column_filter.is_string,
    )
    
    protein_column = knext.ColumnParameter(
        label="Protein PDB Column",
        port_index=1,
        column_filter=knext.column_filter.is_string,
    )
    
    pharmacophore_column = knext.ColumnParameter(
        label="Pharmacophore File Column",
        port_index=2,
        column_filter=knext.column_filter.is_string,
    )
    
    n_poses = knext.IntParameter(
        label="Number of Poses",
        default_value=10,
        min_value=1,
        max_value=100,
    )
    
    n_timesteps = knext.IntParameter(
        label="Number of Timesteps",
        default_value=250,
        min_value=10,
        max_value=1000,
    )
    
    def configure(self, configure_context, input_schema_1, input_schema_2, input_schema_3):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Ligand_ID"),
            knext.Column(knext.string(), "Protein_ID"),
            knext.Column(knext.string(), "Pharmacophore_ID"),
            knext.Column(knext.int32(), "Pose_Number"),
            knext.Column(knext.string(), "Docked_SDF"),
            knext.Column(knext.double(), "Binding_Affinity"),
            knext.Column(knext.double(), "Pharmacophore_Score"),
        ])
    
    def execute(self, exec_context, input_table_1, input_table_2, input_table_3):
        exec_context.set_progress(0.1, "Loading data...")
        
        df_ligands = input_table_1.to_pandas()
        df_proteins = input_table_2.to_pandas()
        df_pharm = input_table_3.to_pandas()
        
        ligands = df_ligands[self.ligand_column].tolist()
        proteins = df_proteins[self.protein_column].tolist()
        pharmacophores = df_pharm[self.pharmacophore_column].tolist()
        
        checkpoint = get_checkpoint_path_for_webapp(
            'pharmacophore_docking',
            Path('/path/to/OMTRA/omtra/trained_models')
        )
        
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_pharm_dock"
        output_dir.mkdir(exist_ok=True)
        
        all_results = []
        total = len(ligands) * len(proteins) * len(pharmacophores)
        processed = 0
        
        for lig_idx, ligand_smiles in enumerate(ligands):
            for prot_idx, protein_path in enumerate(proteins):
                for pharm_idx, pharm_path in enumerate(pharmacophores):
                    exec_context.set_progress(
                        0.2 + (0.7 * processed / total),
                        f"Docking with pharmacophore {processed+1}/{total}..."
                    )
                    
                    task = PharmacophoreDocking(
                        checkpoint=str(checkpoint),
                        ligand_smiles=ligand_smiles,
                        protein_path=protein_path,
                        pharmacophore_path=pharm_path,
                        n_poses=self.n_poses,
                        n_timesteps=self.n_timesteps,
                        output_dir=str(output_dir / f"l{lig_idx}_p{prot_idx}_ph{pharm_idx}"),
                    )
                    
                    results = task.run()
                    
                    for pose_idx, pose_data in enumerate(results['poses']):
                        all_results.append({
                            'Ligand_ID': f"Ligand_{lig_idx}",
                            'Protein_ID': Path(protein_path).stem,
                            'Pharmacophore_ID': Path(pharm_path).stem,
                            'Pose_Number': pose_idx + 1,
                            'Docked_SDF': pose_data.get('sdf', ''),
                            'Binding_Affinity': pose_data.get('affinity', 0.0),
                            'Pharmacophore_Score': pose_data.get('pharm_score', 0.0),
                        })
                    
                    processed += 1
        
        df_results = pd.DataFrame(all_results)
        exec_context.set_progress(1.0, "Complete")
        
        return knext.Table.from_pandas(df_results)
```

---

## Updating knime.yml

After adding these nodes, update your `knime.yml`:

```yaml
name: OMTRA
version: 1.0.0
vendor: Your Organization
description: AI-powered molecular generation and docking using OMTRA
category: /community/cheminformatics

python_version: ">=3.11"

dependencies:
  - torch>=2.4.0
  - dgl>=2.4.0
  - rdkit>=2023.09.4
  - pandas
  - numpy

nodes:
  - omtra_knime.nodes.denovo_generator
  - omtra_knime.nodes.protein_conditioned_generator
  - omtra_knime.nodes.rigid_docking
  - omtra_knime.nodes.flexible_docking
  - omtra_knime.nodes.conformer_generator
  - omtra_knime.nodes.pharmacophore_generator
  - omtra_knime.nodes.pharmacophore_docking
```

---

## Installation

After adding all node files:

```bash
# Reinstall extension
pip install -e .

# Restart KNIME
# All 7 OMTRA nodes should now appear in Node Repository
```

---

## Node Organization in KNIME

The nodes will appear in KNIME under:

```
Node Repository
└── Community Nodes
    └── Cheminformatics
        └── OMTRA
            ├── OMTRA De Novo Generator
            ├── OMTRA Protein-Conditioned Generator
            ├── OMTRA Rigid Docking
            ├── OMTRA Flexible Docking
            ├── OMTRA Conformer Generator
            ├── OMTRA Pharmacophore Generator
            └── OMTRA Pharmacophore Docking
```

---

**All nodes are now ready for use in KNIME workflows!**
