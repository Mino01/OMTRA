# OMTRA KNIME Integration - Quick Start Guide

**Author:** Manus AI  
**Date:** December 11, 2025

---

## Overview

This quick start guide provides the fastest path to getting OMTRA running as KNIME nodes. Follow these steps sequentially for a working integration in under 30 minutes.

---

## Prerequisites Checklist

- ✅ Ubuntu 22.04 or similar Linux system
- ✅ NVIDIA GPU with CUDA 12.1
- ✅ Python 3.11 installed
- ✅ Conda/Miniconda installed
- ✅ KNIME Analytics Platform 5.x installed
- ✅ OMTRA repository cloned

---

## 5-Step Quick Setup

### Step 1: Create Conda Environment (5 minutes)

```bash
# Create environment
conda create -n omtra-knime python=3.11 -y
conda activate omtra-knime

# Install KNIME Python API
pip install knime-extension

# Install OMTRA
cd /path/to/OMTRA
pip install uv
uv pip install -r requirements-cuda.txt --system
uv pip install -e . --system

# Install additional dependencies
pip install pandas pyarrow rdkit
```

### Step 2: Download Model Checkpoints (10 minutes)

```bash
cd /path/to/OMTRA
mkdir -p omtra/trained_models

# Download all checkpoints
wget -r -np -nH --cut-dirs=3 -R "index.html*" \
  -P omtra/trained_models \
  https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/
```

### Step 3: Create Extension Structure (2 minutes)

```bash
# Create project directory
mkdir -p omtra-knime-extension/src/omtra_knime/nodes
cd omtra-knime-extension

# Create __init__.py files
touch src/omtra_knime/__init__.py
touch src/omtra_knime/nodes/__init__.py
```

### Step 4: Create Extension Files (5 minutes)

**Create `knime.yml`:**

```yaml
name: OMTRA
version: 1.0.0
vendor: Your Organization
description: AI-powered molecular generation using OMTRA
category: /community/cheminformatics

python_version: ">=3.11"

nodes:
  - omtra_knime.nodes.denovo_generator
```

**Create `src/omtra_knime/extension.py`:**

```python
import knime.extension as knext

omtra_extension = knext.Extension(
    name="OMTRA",
    version="1.0.0",
    vendor="Your Organization",
    category=knext.category.CHEMINFORMATICS,
)
```

**Create `src/omtra_knime/nodes/denovo_generator.py`:**

```python
import knime.extension as knext
import pandas as pd
import sys
from pathlib import Path

# Add OMTRA to path - UPDATE THIS PATH
sys.path.append('/path/to/OMTRA')

from omtra.tasks.denovo_ligand_condensed import DeNovoLigandCondensed
from omtra.utils.checkpoints import get_checkpoint_path_for_webapp

@knext.node(
    name="OMTRA De Novo Generator",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=None,
    category="/community/cheminformatics/omtra",
)
@knext.output_table(
    name="Generated Molecules",
    description="Generated molecules with SMILES and properties"
)
class OMTRADeNovoGenerator:
    """Generate novel molecules using OMTRA."""
    
    n_samples = knext.IntParameter(
        label="Number of Samples",
        description="Number of molecules to generate",
        default_value=10,
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
    
    def configure(self, configure_context):
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "SMILES"),
            knext.Column(knext.double(), "Molecular_Weight"),
        ])
    
    def execute(self, exec_context):
        exec_context.set_progress(0.1, "Loading model...")
        
        # Get checkpoint - UPDATE THIS PATH
        checkpoint = get_checkpoint_path_for_webapp(
            'denovo_ligand_condensed',
            Path('/path/to/OMTRA/omtra/trained_models')
        )
        
        # Create output directory
        output_dir = Path(exec_context.get_workflow_temp_dir()) / "omtra_output"
        output_dir.mkdir(exist_ok=True)
        
        exec_context.set_progress(0.3, "Generating molecules...")
        
        # Run OMTRA
        task = DeNovoLigandCondensed(
            checkpoint=str(checkpoint),
            n_samples=self.n_samples,
            n_timesteps=self.n_timesteps,
            output_dir=str(output_dir),
        )
        
        results = task.run()
        
        exec_context.set_progress(0.8, "Processing results...")
        
        # Convert to DataFrame
        molecules_data = []
        for mol_data in results['molecules']:
            molecules_data.append({
                'SMILES': mol_data.get('smiles', ''),
                'Molecular_Weight': mol_data.get('molecular_weight', 0.0),
            })
        
        df = pd.DataFrame(molecules_data)
        exec_context.set_progress(1.0, "Complete")
        
        return knext.Table.from_pandas(df)
```

**Create `setup.py`:**

```python
from setuptools import setup, find_packages

setup(
    name='omtra-knime-extension',
    version='1.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['knime-extension', 'pandas'],
    python_requires='>=3.11',
)
```

### Step 5: Install and Test (5 minutes)

```bash
# Install extension
pip install -e .

# Configure KNIME
# 1. Open KNIME
# 2. Go to File → Preferences → KNIME → Python
# 3. Select "Conda" and point to omtra-knime environment
# 4. Test connection
# 5. Restart KNIME

# Look for "OMTRA De Novo Generator" in Node Repository
```

---

## Quick Test Workflow

1. Open KNIME
2. Create new workflow
3. Search for "OMTRA De Novo Generator"
4. Drag node to workflow
5. Configure: n_samples=5, n_timesteps=50
6. Add "Table View" node after OMTRA node
7. Execute workflow
8. View generated molecules

---

## Troubleshooting Quick Fixes

**Node doesn't appear:**
```bash
# Verify installation
conda activate omtra-knime
pip list | grep omtra-knime

# Check KNIME Python config
# File → Preferences → KNIME → Python
# Ensure omtra-knime environment is selected
```

**Import errors:**
```bash
# Verify OMTRA installation
conda activate omtra-knime
python -c "import omtra; print('OK')"

# Reinstall if needed
cd /path/to/OMTRA
pip install -e .
```

**CUDA errors:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch==2.4.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Next Steps

After successful setup:

1. **Add More Nodes** - Implement protein-conditioned, docking, and conformer nodes
2. **Customize Parameters** - Add advanced settings for fine-tuning
3. **Create Workflows** - Build complete drug discovery pipelines
4. **Share Extension** - Package and distribute to your team

---

## Complete Example Repository Structure

```
omtra-knime-extension/
├── knime.yml
├── setup.py
├── README.md
└── src/
    └── omtra_knime/
        ├── __init__.py
        ├── extension.py
        └── nodes/
            ├── __init__.py
            └── denovo_generator.py
```

---

## Support

For detailed documentation, see `OMTRA_KNIME_Integration_Guide.md`.

For issues:
- Check KNIME log: View → Open KNIME Log
- Verify Python environment configuration
- Ensure OMTRA checkpoints are downloaded
- Test OMTRA CLI independently first

---

**You're now ready to use OMTRA in KNIME!** 🎉
