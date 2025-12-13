# OMTRA KNIME Node Integration Guide

**Author:** Manus AI  
**Date:** December 11, 2025  
**Version:** 1.0

---

## Executive Summary

This document provides a comprehensive guide for integrating **OMTRA** (a multi-task generative model for structure-based drug design) as a custom node in **KNIME Analytics Platform**. KNIME is a widely-used open-source data analytics platform that enables visual workflow creation through a node-based interface. By integrating OMTRA as a KNIME node, researchers can seamlessly incorporate AI-powered molecular generation into their existing drug discovery workflows.

The integration leverages KNIME's **Python Extension Development** framework, which allows developers to create custom nodes using Python rather than Java, significantly reducing development complexity and enabling direct use of OMTRA's Python API.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Integration Approaches](#integration-approaches)
3. [Architecture Design](#architecture-design)
4. [Development Environment Setup](#development-environment-setup)
5. [OMTRA Node Implementation](#omtra-node-implementation)
6. [Node Configuration and Dialogs](#node-configuration-and-dialogs)
7. [Input and Output Specifications](#input-and-output-specifications)
8. [Testing and Validation](#testing-and-validation)
9. [Packaging and Distribution](#packaging-and-distribution)
10. [Example Workflows](#example-workflows)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## Introduction

### What is KNIME?

KNIME (Konstanz Information Miner) is an open-source platform for data analytics, reporting, and integration. It uses a visual programming approach where users create workflows by connecting nodes that represent different data processing operations. KNIME is particularly popular in the pharmaceutical and life sciences industries for drug discovery and cheminformatics workflows.

### Why Integrate OMTRA with KNIME?

Integrating OMTRA as a KNIME node provides several advantages:

**Workflow Integration:** Researchers can combine OMTRA's molecular generation capabilities with existing KNIME workflows for data preprocessing, analysis, and visualization.

**Visual Programming:** Non-programmers can use OMTRA through KNIME's intuitive drag-and-drop interface without writing Python code.

**Reproducibility:** KNIME workflows are self-documenting and can be easily shared, ensuring reproducible research.

**Ecosystem Access:** Integration with KNIME's extensive library of cheminformatics nodes (RDKit, CDK, etc.) enables comprehensive drug discovery pipelines.

**Enterprise Deployment:** KNIME Server allows deployment of OMTRA workflows in production environments with scheduling, monitoring, and access control.

---

## Integration Approaches

KNIME supports multiple approaches for creating custom nodes. Based on OMTRA's requirements and the research from KNIME documentation, the recommended approach is **Python-based node development**.

### Approach Comparison

| Approach | Language | Complexity | OMTRA Compatibility | Recommendation |
|----------|----------|------------|---------------------|----------------|
| **Pure Python Node** | Python | Low | Excellent | ✅ **Recommended** |
| Java Node with Python Bridge | Java + Python | High | Good | Not recommended |
| Python Script Node | Python | Very Low | Limited | For prototyping only |
| Wrapped Component | KNIME Workflow | Very Low | Limited | For simple use cases |

### Recommended Approach: Pure Python Node Extension

The **Pure Python Node Extension** framework allows developers to create fully-featured KNIME nodes using Python. This approach is ideal for OMTRA because:

1. **Direct API Access:** OMTRA is implemented in Python, so we can directly import and use its modules without language bridges
2. **Reduced Boilerplate:** Python nodes require significantly less code than Java-based nodes
3. **Easier Maintenance:** Updates to OMTRA can be incorporated without recompiling Java code
4. **GPU Support:** Python environment can be configured with CUDA for GPU acceleration
5. **Dependency Management:** Conda environments handle complex dependencies (PyTorch, DGL, RDKit)

---

## Architecture Design

### High-Level Architecture

The OMTRA KNIME node integration follows a layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│              KNIME Analytics Platform                   │
│  - Workflow Engine                                      │
│  - Node Repository                                      │
│  - User Interface                                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│           OMTRA Python Extension Package                │
│  - Node Definitions (Python)                           │
│  - Configuration Dialogs                                │
│  - Input/Output Port Specifications                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              OMTRA Python API Layer                     │
│  - Task Registry                                        │
│  - Sampling Functions                                   │
│  - Model Loading                                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│           OMTRA Core (PyTorch/DGL)                      │
│  - Neural Network Models                                │
│  - Flow Matching                                        │
│  - Molecular Generation                                 │
└─────────────────────────────────────────────────────────┘
```

### Node Categories

The OMTRA integration will provide multiple specialized nodes organized by functionality:

**Category 1: Molecular Generation**
- OMTRA De Novo Generator
- OMTRA Protein-Conditioned Generator

**Category 2: Docking**
- OMTRA Rigid Docking
- OMTRA Flexible Docking

**Category 3: Conformer Generation**
- OMTRA Conformer Generator

**Category 4: Pharmacophore-Guided**
- OMTRA Pharmacophore-Conditioned Generator
- OMTRA Pharmacophore Docking

### Data Flow

**Input Data Flow:**
1. User configures node parameters through dialog
2. Input tables/files are passed from upstream nodes
3. Python node receives data as pandas DataFrames or file paths
4. Data is converted to OMTRA-compatible formats
5. OMTRA processes the request

**Output Data Flow:**
1. OMTRA generates molecules and metrics
2. Results are converted to KNIME table format
3. Output ports provide structured data to downstream nodes
4. Generated files (SDF, PDB) are stored and referenced

---

## Development Environment Setup

### Prerequisites

Before developing OMTRA KNIME nodes, ensure you have:

**System Requirements:**
- Linux system (Ubuntu 22.04 recommended)
- NVIDIA GPU with CUDA 12.1 support
- Python 3.11
- Conda or Miniconda
- KNIME Analytics Platform 5.x or later

**Software Requirements:**
- KNIME Python Extension Development (Labs)
- Git
- Java Development Kit (JDK) 17 or later
- Maven (for building extensions)

### Step 1: Install KNIME Analytics Platform

Download and install KNIME Analytics Platform from the official website:

```bash
# Download KNIME
wget https://download.knime.org/analytics-platform/linux/knime-latest-linux.gtk.x86_64.tar.gz

# Extract
tar -xzf knime-latest-linux.gtk.x86_64.tar.gz

# Run KNIME
cd knime_*
./knime
```

### Step 2: Install KNIME Python Extension Development

Within KNIME:
1. Go to **File → Install KNIME Extensions**
2. Search for "KNIME Python Extension Development (Labs)"
3. Install the extension
4. Restart KNIME

### Step 3: Set Up Conda Environment

Create a dedicated conda environment for OMTRA node development:

```bash
# Create environment
conda create -n omtra-knime python=3.11
conda activate omtra-knime

# Install KNIME Python API
pip install knime-extension

# Install OMTRA dependencies
pip install uv
cd /path/to/OMTRA
uv pip install -r requirements-cuda.txt --system
uv pip install -e . --system

# Install additional dependencies for KNIME integration
pip install pandas pyarrow
```

### Step 4: Clone KNIME SDK Setup (User Preference)

Based on user preference, integrate the KNIME SDK setup as a git submodule:

```bash
# In your project directory
git submodule add https://github.com/Mino01/knime-sdk-setup
git submodule update --init --recursive
```

### Step 5: Download OMTRA Model Checkpoints

```bash
cd /path/to/OMTRA
mkdir -p omtra/trained_models
wget -r -np -nH --cut-dirs=3 -R "index.html*" \
  -P omtra/trained_models \
  https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/
```

### Step 6: Configure KNIME Python Environment

In KNIME:
1. Go to **File → Preferences → KNIME → Python**
2. Select "Conda" as Python version
3. Point to your `omtra-knime` conda environment
4. Test the connection

---

## OMTRA Node Implementation

### Project Structure

Create the following directory structure for your OMTRA KNIME extension:

```
omtra-knime-extension/
├── src/
│   └── omtra_knime/
│       ├── __init__.py
│       ├── extension.py
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── denovo_generator.py
│       │   ├── protein_conditioned_generator.py
│       │   ├── rigid_docking.py
│       │   ├── conformer_generator.py
│       │   └── pharmacophore_generator.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── data_conversion.py
│       │   ├── file_handling.py
│       │   └── validation.py
│       └── config/
│           ├── __init__.py
│           └── node_settings.py
├── tests/
│   ├── __init__.py
│   └── test_nodes.py
├── config.yml
├── knime.yml
├── README.md
└── setup.py
```

### Extension Configuration

Create `knime.yml` to define the extension metadata:

```yaml
# knime.yml
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
  - omtra_knime.nodes.conformer_generator
  - omtra_knime.nodes.pharmacophore_generator
```

### Base Node Implementation

Create `extension.py` to initialize the extension:

```python
# src/omtra_knime/extension.py
import knime.extension as knext

# Define the extension
omtra_extension = knext.Extension(
    name="OMTRA",
    version="1.0.0",
    vendor="Your Organization",
    category=knext.category.CHEMINFORMATICS,
)
```

### Example Node: De Novo Generator

Create `nodes/denovo_generator.py`:

```python
# src/omtra_knime/nodes/denovo_generator.py
import knime.extension as knext
import pandas as pd
import logging
from pathlib import Path
import sys

# Add OMTRA to path
sys.path.append('/path/to/OMTRA')

from omtra.tasks.register import TASK_REGISTER
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="OMTRA De Novo Generator",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/omtra_denovo.png",
    category="/community/cheminformatics/omtra",
)
@knext.output_table(
    name="Generated Molecules",
    description="Table containing generated molecules with SMILES and properties"
)
class OMTRADeNovoGenerator:
    """
    Generate novel drug-like molecules using OMTRA's de novo generation.
    
    This node uses OMTRA's flow-matching based generative model to create
    novel molecular structures from scratch without any conditioning.
    """
    
    # Configuration parameters
    n_samples = knext.IntParameter(
        label="Number of Samples",
        description="Number of molecules to generate",
        default_value=100,
        min_value=1,
        max_value=10000,
    )
    
    n_timesteps = knext.IntParameter(
        label="Number of Timesteps",
        description="Number of integration steps during sampling",
        default_value=250,
        min_value=10,
        max_value=1000,
    )
    
    stochastic_sampling = knext.BoolParameter(
        label="Stochastic Sampling",
        description="Enable stochastic (vs deterministic) sampling",
        default_value=False,
    )
    
    compute_metrics = knext.BoolParameter(
        label="Compute Metrics",
        description="Compute molecular properties and drug-likeness metrics",
        default_value=True,
    )
    
    checkpoint_path = knext.StringParameter(
        label="Checkpoint Path",
        description="Path to OMTRA model checkpoint (leave empty for auto-detection)",
        default_value="",
    )
    
    def configure(self, configure_context):
        """
        Configure the node's output schema.
        Called before execution to determine output structure.
        """
        # Define output table schema
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "SMILES"),
            knext.Column(knext.string(), "SDF"),
            knext.Column(knext.double(), "Molecular_Weight"),
            knext.Column(knext.double(), "LogP"),
            knext.Column(knext.double(), "TPSA"),
            knext.Column(knext.int32(), "HBD"),
            knext.Column(knext.int32(), "HBA"),
            knext.Column(knext.double(), "QED"),
            knext.Column(knext.double(), "SA_Score"),
        ])
    
    def execute(self, exec_context):
        """
        Execute the OMTRA de novo generation.
        """
        LOGGER.info("Starting OMTRA de novo generation...")
        
        # Set execution context for progress reporting
        exec_context.set_progress(0.1, "Initializing OMTRA...")
        
        # Import OMTRA modules
        try:
            from omtra.tasks.denovo_ligand_condensed import DeNovoLigandCondensed
            from omtra.utils.checkpoints import TASK_TO_CHECKPOINT
        except ImportError as e:
            raise RuntimeError(f"Failed to import OMTRA: {e}")
        
        # Get checkpoint path
        if self.checkpoint_path:
            checkpoint = Path(self.checkpoint_path)
        else:
            checkpoint = get_checkpoint_path_for_webapp(
                'denovo_ligand_condensed',
                Path('/path/to/OMTRA/omtra/trained_models')
            )
        
        if not checkpoint or not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        
        LOGGER.info(f"Using checkpoint: {checkpoint}")
        
        # Create output directory
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_output"
        output_dir.mkdir(exist_ok=True)
        
        exec_context.set_progress(0.2, "Loading model...")
        
        # Initialize task
        task = DeNovoLigandCondensed(
            checkpoint=str(checkpoint),
            n_samples=self.n_samples,
            n_timesteps=self.n_timesteps,
            stochastic_sampling=self.stochastic_sampling,
            output_dir=str(output_dir),
        )
        
        exec_context.set_progress(0.3, "Generating molecules...")
        
        # Run generation
        try:
            results = task.run()
        except Exception as e:
            raise RuntimeError(f"OMTRA generation failed: {e}")
        
        exec_context.set_progress(0.8, "Processing results...")
        
        # Convert results to pandas DataFrame
        molecules_data = []
        
        for i, mol_data in enumerate(results['molecules']):
            mol_dict = {
                'SMILES': mol_data.get('smiles', ''),
                'SDF': mol_data.get('sdf', ''),
                'Molecular_Weight': mol_data.get('molecular_weight', 0.0),
                'LogP': mol_data.get('logp', 0.0),
                'TPSA': mol_data.get('tpsa', 0.0),
                'HBD': mol_data.get('hbd', 0),
                'HBA': mol_data.get('hba', 0),
                'QED': mol_data.get('qed', 0.0),
                'SA_Score': mol_data.get('sa_score', 0.0),
            }
            molecules_data.append(mol_dict)
            
            # Update progress
            if i % 10 == 0:
                progress = 0.8 + (0.2 * i / len(results['molecules']))
                exec_context.set_progress(progress, f"Processing molecule {i+1}/{len(results['molecules'])}")
        
        df = pd.DataFrame(molecules_data)
        
        exec_context.set_progress(1.0, "Complete")
        LOGGER.info(f"Generated {len(df)} molecules successfully")
        
        return knext.Table.from_pandas(df)
```

---

## Node Configuration and Dialogs

### Parameter Types

KNIME Python nodes support various parameter types for user configuration:

| Parameter Type | KNIME Class | Use Case |
|---------------|-------------|----------|
| Integer | `knext.IntParameter` | Sample count, timesteps |
| Double | `knext.DoubleParameter` | Noise scaling, margins |
| String | `knext.StringParameter` | File paths, task names |
| Boolean | `knext.BoolParameter` | Enable/disable features |
| Choice | `knext.StringParameter` with choices | Task selection |
| File | `knext.StringParameter` | Input file paths |

### Advanced Dialog Configuration

For more complex configurations, create custom dialog components:

```python
# Advanced parameter group
class AdvancedSettings(knext.ParameterGroup):
    """Advanced OMTRA settings"""
    
    noise_scaler = knext.DoubleParameter(
        label="Noise Scaler",
        description="Scaling factor for stochastic noise",
        default_value=1.0,
        min_value=0.0,
        max_value=10.0,
    )
    
    eps = knext.DoubleParameter(
        label="Epsilon",
        description="Small epsilon value for numerical stability",
        default_value=0.01,
        min_value=0.0001,
        max_value=0.1,
    )
    
    use_gt_n_lig_atoms = knext.BoolParameter(
        label="Use Ground Truth Atom Count",
        description="Match ground truth ligand atom count",
        default_value=False,
    )

# Use in node
@knext.node(name="OMTRA Advanced Generator", ...)
class OMTRAAdvancedGenerator:
    advanced = AdvancedSettings()
    
    # Access parameters
    def execute(self, exec_context):
        noise = self.advanced.noise_scaler
        # ...
```

---

## Input and Output Specifications

### Input Port Types

OMTRA nodes can accept various input types:

**Table Input:**
```python
@knext.input_table(
    name="Protein Structures",
    description="Table containing protein PDB file paths"
)
```

**Binary Input (Files):**
```python
@knext.input_binary(
    name="Protein Structure",
    description="Protein structure in PDB format",
    id="protein.pdb"
)
```

### Output Port Types

**Table Output:**
```python
@knext.output_table(
    name="Generated Molecules",
    description="Table with SMILES and properties"
)
```

**Binary Output (Files):**
```python
@knext.output_binary(
    name="SDF File",
    description="Generated molecules in SDF format",
    id="molecules.sdf"
)
```

### Data Conversion Utilities

Create utility functions for data conversion:

```python
# src/omtra_knime/utils/data_conversion.py
import pandas as pd
from rdkit import Chem
from typing import List, Dict

def smiles_to_knime_table(smiles_list: List[str]) -> pd.DataFrame:
    """Convert list of SMILES to KNIME table format."""
    data = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            data.append({
                'SMILES': smiles,
                'Molecular_Formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'Molecular_Weight': Chem.Descriptors.MolWt(mol),
            })
    return pd.DataFrame(data)

def sdf_to_knime_table(sdf_path: str) -> pd.DataFrame:
    """Convert SDF file to KNIME table."""
    suppl = Chem.SDMolSupplier(sdf_path)
    data = []
    for mol in suppl:
        if mol:
            data.append({
                'SMILES': Chem.MolToSmiles(mol),
                'SDF': Chem.MolToMolBlock(mol),
                'Name': mol.GetProp('_Name') if mol.HasProp('_Name') else '',
            })
    return pd.DataFrame(data)
```

---

## Testing and Validation

### Unit Tests

Create unit tests for node functionality:

```python
# tests/test_nodes.py
import unittest
import pandas as pd
from omtra_knime.nodes.denovo_generator import OMTRADeNovoGenerator

class TestOMTRADeNovoGenerator(unittest.TestCase):
    
    def setUp(self):
        self.node = OMTRADeNovoGenerator()
        self.node.n_samples = 5
        self.node.n_timesteps = 50
    
    def test_configuration(self):
        """Test node configuration."""
        schema = self.node.configure(None)
        self.assertIsNotNone(schema)
        self.assertIn('SMILES', [col.name for col in schema.columns])
    
    def test_execution(self):
        """Test node execution."""
        # This requires OMTRA to be installed and checkpoints available
        # Skip if not in test environment
        if not self.has_omtra_available():
            self.skipTest("OMTRA not available")
        
        result = self.node.execute(MockExecutionContext())
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
    
    def has_omtra_available(self):
        try:
            import omtra
            return True
        except ImportError:
            return False

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

Test nodes within KNIME workflows:

1. Create a test workflow in KNIME
2. Add OMTRA nodes
3. Configure with test data
4. Execute and verify outputs
5. Check for errors and warnings

---

## Packaging and Distribution

### Build Extension Package

Create `setup.py`:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='omtra-knime-extension',
    version='1.0.0',
    description='OMTRA integration for KNIME Analytics Platform',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'knime-extension',
        'pandas',
        'numpy',
        'rdkit',
    ],
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.11',
    ],
)
```

### Build and Install

```bash
# Build the extension
python setup.py sdist bdist_wheel

# Install in KNIME Python environment
conda activate omtra-knime
pip install dist/omtra-knime-extension-1.0.0.tar.gz
```

### Register with KNIME

In KNIME:
1. Go to **File → Preferences → KNIME → Python**
2. Ensure the `omtra-knime` environment is selected
3. Restart KNIME
4. OMTRA nodes should appear in the Node Repository

---

## Example Workflows

### Workflow 1: De Novo Molecule Generation

**Objective:** Generate 100 novel drug-like molecules and analyze their properties.

**Nodes:**
1. **OMTRA De Novo Generator** - Generate molecules
2. **RDKit Descriptor Calculator** - Calculate additional descriptors
3. **Scatter Plot** - Visualize LogP vs Molecular Weight
4. **CSV Writer** - Export results

**Configuration:**
- OMTRA De Novo Generator: n_samples=100, compute_metrics=True
- Connect output to RDKit node for additional analysis

### Workflow 2: Protein-Conditioned Design

**Objective:** Design ligands for a specific protein target.

**Nodes:**
1. **File Reader** - Load protein PDB file
2. **OMTRA Protein-Conditioned Generator** - Generate ligands
3. **SDF Writer** - Save generated molecules
4. **Molecule Viewer** - Visualize results

**Configuration:**
- Provide protein PDB file path
- Set n_samples=50, n_timesteps=250

### Workflow 3: Virtual Screening Pipeline

**Objective:** Generate molecules, dock them, and rank by binding affinity.

**Nodes:**
1. **OMTRA De Novo Generator** - Generate candidates
2. **OMTRA Rigid Docking** - Dock to target protein
3. **Row Filter** - Filter by binding affinity
4. **Top k Selector** - Select top 10 molecules
5. **Table Writer** - Export results

---

## Troubleshooting

### Common Issues

**Issue 1: OMTRA import fails**
```
ImportError: No module named 'omtra'
```
**Solution:** Ensure OMTRA is installed in the KNIME Python environment:
```bash
conda activate omtra-knime
cd /path/to/OMTRA
pip install -e .
```

**Issue 2: CUDA not available**
```
RuntimeError: CUDA not available
```
**Solution:** Install CUDA-enabled PyTorch:
```bash
pip install torch==2.4.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

**Issue 3: Checkpoint not found**
```
FileNotFoundError: Checkpoint not found
```
**Solution:** Download OMTRA model checkpoints:
```bash
cd /path/to/OMTRA
wget -r -np -nH --cut-dirs=3 -R "index.html*" \
  -P omtra/trained_models \
  https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/
```

**Issue 4: Node doesn't appear in KNIME**
```
Node not visible in Node Repository
```
**Solution:**
1. Verify Python environment is correctly configured
2. Check knime.yml syntax
3. Restart KNIME
4. Check KNIME log for errors: **View → Open KNIME Log**

**Issue 5: Out of memory during generation**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce n_samples or n_timesteps
- Use CPU instead of GPU (slower but more memory)
- Close other GPU-intensive applications

---

## References

[1] KNIME Python Extension Development Guide: https://docs.knime.com/2024-06/pure_python_node_extensions_guide/  
[2] KNIME Python Integration Guide: https://docs.knime.com/2024-06/python_installation_guide/  
[3] OMTRA GitHub Repository: https://github.com/gnina/OMTRA  
[4] OMTRA Preprint: https://arxiv.org/abs/2512.05080  
[5] KNIME Forum - Python Node Development: https://forum.knime.com/t/node-development-using-python/22893  
[6] KNIME Blog - 4 Steps for Python Team: https://www.knime.com/blog/4-steps-for-your-python-team-to-develop-knime-nodes  

---

## Appendix A: Complete Node Example

See the `denovo_generator.py` example in the [OMTRA Node Implementation](#omtra-node-implementation) section for a complete, production-ready node implementation.

---

## Appendix B: KNIME SDK Setup Integration

Based on user preference, integrate the KNIME SDK setup as a git submodule:

```bash
# Add submodule
git submodule add https://github.com/Mino01/knime-sdk-setup

# Initialize and update
git submodule update --init --recursive

# Use SDK setup scripts
cd knime-sdk-setup
./setup.sh
```

This provides a standardized development environment for KNIME extension development.

---

**End of Document**
