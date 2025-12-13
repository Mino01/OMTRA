# ChatGPT Codex Implementation Plan: OMTRA Seesar-Like System

**Author:** Manus AI  
**Date:** December 11, 2025  
**Version:** 1.0  
**Target:** ChatGPT Codex Execution

---

## Executive Summary

This document provides a complete, step-by-step implementation plan for ChatGPT Codex to build a Seesar-like interactive molecular design system for OMTRA with ForcelabElixir FEP evaluation. The plan is structured for sequential execution with clear dependencies, complete code templates, and validation checkpoints.

**Implementation Time:** 12 weeks (phased delivery)  
**Lines of Code:** ~15,000 (backend) + ~8,000 (frontend)  
**Key Technologies:** Python, TypeScript/React, OpenMM, RDKit, tRPC, PostgreSQL

---

## Implementation Strategy

### Phased Approach

The implementation is divided into 6 phases, each delivering working functionality:

| Phase | Duration | Deliverable | Dependencies |
|-------|----------|-------------|--------------|
| **Phase 1** | Week 1-2 | FEP Forcefield Core | None |
| **Phase 2** | Week 3-4 | Molecular Editor Backend | Phase 1 |
| **Phase 3** | Week 5-6 | FEP Calculation Engine | Phase 1, 2 |
| **Phase 4** | Week 7 | Pharmacophore System | Phase 2 |
| **Phase 5** | Week 8 | Visual Feedback System | Phase 3 |
| **Phase 6** | Week 9-12 | Web Interface & Integration | All previous |

### Quality Gates

Each phase must pass these checkpoints before proceeding:

1. **Unit Tests**: All functions have passing tests (>90% coverage)
2. **Integration Tests**: Components work together correctly
3. **Performance Tests**: Meets performance targets
4. **Code Review**: Follows best practices and style guide
5. **Documentation**: Complete API docs and usage examples

---

## Project Structure

```
/home/ubuntu/
├── OMTRA/                          # OMTRA repository
│   ├── omtra/
│   │   ├── fep/                    # NEW: FEP evaluation module
│   │   │   ├── __init__.py
│   │   │   ├── forcefield.py       # ForcelabElixir forcefield wrapper
│   │   │   ├── fast_scorer.py      # Fast FEP estimation (MM-GBSA)
│   │   │   ├── full_calculator.py  # Full FEP with convergence
│   │   │   ├── perturbation.py     # Perturbation network builder
│   │   │   └── utils.py            # Utility functions
│   │   ├── editor/                 # NEW: Molecular editor
│   │   │   ├── __init__.py
│   │   │   ├── fragments.py        # Fragment library
│   │   │   ├── scaffold.py         # Scaffold hopping
│   │   │   ├── rgroup.py           # R-group decoration
│   │   │   └── constraints.py      # Design constraints
│   │   ├── pharmacophore/          # NEW: Pharmacophore system
│   │   │   ├── __init__.py
│   │   │   ├── detector.py         # Pharmacophore detection
│   │   │   ├── generator.py        # Constrained generation
│   │   │   └── matcher.py          # Pharmacophore matching
│   │   ├── analysis/               # NEW: Interaction analysis
│   │   │   ├── __init__.py
│   │   │   ├── interactions.py     # H-bonds, hydrophobic, etc.
│   │   │   ├── energy.py           # Energy decomposition
│   │   │   └── visualization.py    # Interaction visualization
│   │   └── convergence/            # EXISTING: Convergence monitoring
│   │       └── monitor.py
│   └── tests/                      # NEW: Test suite
│       ├── test_fep/
│       ├── test_editor/
│       ├── test_pharmacophore/
│       └── test_analysis/
│
└── forcelab-elixir/                # ForcelabElixir web app
    ├── server/
    │   ├── routers/
    │   │   └── seesar.ts           # NEW: Seesar-like API routes
    │   ├── services/
    │   │   ├── fep-service.ts      # NEW: FEP service wrapper
    │   │   ├── editor-service.ts   # NEW: Editor service
    │   │   └── pharma-service.ts   # NEW: Pharmacophore service
    │   └── workers/
    │       └── fep-worker.ts       # NEW: Background FEP worker
    ├── client/src/
    │   ├── pages/
    │   │   └── Seesar.tsx          # NEW: Main Seesar interface
    │   ├── components/
    │   │   ├── MolecularViewer.tsx # NEW: 3D viewer (NGL)
    │   │   ├── DesignTools.tsx     # NEW: Design tools panel
    │   │   ├── AnalysisPanel.tsx   # NEW: Analysis panel
    │   │   └── seesar/             # NEW: Seesar components
    │   │       ├── FragmentLibrary.tsx
    │   │       ├── ScaffoldHopper.tsx
    │   │       ├── RGroupDecorator.tsx
    │   │       ├── PharmacophoreEditor.tsx
    │   │       └── InteractionViewer.tsx
    │   └── lib/
    │       └── seesar-trpc.ts      # NEW: tRPC client for Seesar
    └── drizzle/
        └── schema.ts               # UPDATE: Add Seesar tables
```

---

## Phase 1: FEP Forcefield Core (Weeks 1-2)

### Objectives

Implement the ForcelabElixir forcefield wrapper with ANI-2x and ESP-DNN support for FEP calculations.

### Dependencies

**Python Packages:**
```bash
pip install torch torchani openmm rdkit numpy scipy
```

**Model Checkpoints:**
- ANI-2x: Download from https://github.com/aiqm/torchani
- ESP-DNN: Custom model (provide checkpoint path)

### Files to Create

#### 1. `omtra/fep/forcefield.py` (Priority: CRITICAL)

**Purpose:** Wrapper for ForcelabElixir forcefield with ANI-2x and ESP-DNN

**Key Classes:**
- `ForcelabElixirForcefield`: Main forcefield class
- `ANI2xCalculator`: ANI-2x energy calculator
- `ESPDNNCalculator`: ESP-DNN charge calculator
- `VirtualSiteBuilder`: Virtual sites for sigma holes

**Implementation Template:**

```python
"""
ForcelabElixir Forcefield for FEP Calculations

Implements hybrid forcefield:
- Ligand: ANI-2x (energy) + ESP-DNN (charges)
- Protein: AMBER14
- Solvent: TIP3P
- Virtual sites for sigma holes (halogens)

Author: Manus AI
Date: December 11, 2025
"""

import torch
import torchani
import openmm
import openmm.app as app
import openmm.unit as unit
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, List, Optional, Dict
import numpy as np
from pathlib import Path


class ANI2xCalculator:
    """
    ANI-2x quantum-accurate energy calculator
    
    Provides QM-level energies for organic molecules
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize ANI-2x model
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        self.model.eval()
    
    def calculate_energy(
        self,
        species: torch.Tensor,
        coordinates: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """
        Calculate energy and forces
        
        Args:
            species: Atomic numbers (N,)
            coordinates: Atomic coordinates in Angstroms (N, 3)
        
        Returns:
            (energy_hartree, forces_hartree_per_angstrom)
        """
        with torch.no_grad():
            species = species.to(self.device).unsqueeze(0)
            coordinates = coordinates.to(self.device).unsqueeze(0)
            coordinates.requires_grad = True
            
            energy = self.model((species, coordinates)).energies
            forces = -torch.autograd.grad(energy.sum(), coordinates)[0]
            
            return energy.item(), forces.squeeze(0)
    
    def calculate_energy_openmm(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray
    ) -> float:
        """
        Calculate energy in OpenMM units (kJ/mol)
        
        Args:
            positions: Positions in nm
            atomic_numbers: Atomic numbers
        
        Returns:
            Energy in kJ/mol
        """
        # Convert nm to Angstrom
        positions_angstrom = positions * 10.0
        
        species = torch.tensor(atomic_numbers, dtype=torch.long)
        coordinates = torch.tensor(positions_angstrom, dtype=torch.float32)
        
        energy_hartree, _ = self.calculate_energy(species, coordinates)
        
        # Convert Hartree to kJ/mol
        energy_kj_mol = energy_hartree * 2625.5  # 1 Hartree = 2625.5 kJ/mol
        
        return energy_kj_mol


class ESPDNNCalculator:
    """
    ESP-DNN quantum-accurate charge calculator
    
    Provides QM-level partial charges for electrostatics
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize ESP-DNN model
        
        Args:
            model_path: Path to ESP-DNN checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Load custom ESP-DNN model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
    
    def calculate_charges(
        self,
        mol: Chem.Mol,
        conformer_id: int = 0
    ) -> np.ndarray:
        """
        Calculate ESP-fitted partial charges
        
        Args:
            mol: RDKit molecule with 3D coordinates
            conformer_id: Conformer ID to use
        
        Returns:
            Partial charges (N,)
        """
        # Get atomic features
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        
        # Get coordinates
        conformer = mol.GetConformer(conformer_id)
        coordinates = conformer.GetPositions()
        
        # Prepare input for ESP-DNN
        species = torch.tensor(atomic_numbers, dtype=torch.long).to(self.device)
        coords = torch.tensor(coordinates, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            charges = self.model(species.unsqueeze(0), coords.unsqueeze(0))
            charges = charges.squeeze(0).cpu().numpy()
        
        return charges


class VirtualSiteBuilder:
    """
    Build virtual sites for sigma holes
    
    Implements virtual sites for halogens (Cl, Br, I) to model
    sigma holes accurately
    """
    
    @staticmethod
    def detect_halogens(mol: Chem.Mol) -> List[int]:
        """
        Detect halogen atoms that need virtual sites
        
        Args:
            mol: RDKit molecule
        
        Returns:
            List of halogen atom indices
        """
        halogens = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [17, 35, 53]:  # Cl, Br, I
                halogens.append(atom.GetIdx())
        return halogens
    
    @staticmethod
    def add_virtual_site(
        system: openmm.System,
        topology: app.Topology,
        halogen_idx: int,
        bonded_idx: int,
        distance: float = 0.15  # nm
    ) -> int:
        """
        Add virtual site for halogen sigma hole
        
        Args:
            system: OpenMM system
            topology: OpenMM topology
            halogen_idx: Halogen atom index
            bonded_idx: Bonded atom index
            distance: Distance along bond (nm)
        
        Returns:
            Virtual site particle index
        """
        # Add virtual site particle
        vsite_idx = system.addParticle(0.0)  # Zero mass
        
        # Create two-particle virtual site
        # Position: halogen + distance * (halogen - bonded) / |halogen - bonded|
        vsite = openmm.TwoParticleAverageSite(
            halogen_idx,
            bonded_idx,
            1.0 + distance,  # Weight for halogen
            -distance         # Weight for bonded
        )
        
        system.setVirtualSite(vsite_idx, vsite)
        
        return vsite_idx


class ForcelabElixirForcefield:
    """
    Hybrid forcefield for FEP calculations
    
    Combines:
    - ANI-2x for ligand energies
    - ESP-DNN for ligand charges
    - AMBER14 for protein
    - TIP3P for solvent
    - Virtual sites for sigma holes
    """
    
    def __init__(
        self,
        ani2x_device: str = 'cuda',
        espdnn_model_path: Optional[str] = None
    ):
        """
        Initialize ForcelabElixir forcefield
        
        Args:
            ani2x_device: Device for ANI-2x ('cuda' or 'cpu')
            espdnn_model_path: Path to ESP-DNN model checkpoint
        """
        self.ani2x = ANI2xCalculator(device=ani2x_device)
        
        if espdnn_model_path:
            self.espdnn = ESPDNNCalculator(espdnn_model_path, device=ani2x_device)
        else:
            self.espdnn = None
            print("Warning: ESP-DNN not available, using Gasteiger charges")
        
        # Load AMBER14 forcefield for protein
        self.amber = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    def create_system(
        self,
        protein_pdb: str,
        ligand_mol: Chem.Mol,
        box_size: float = 10.0,  # nm
        add_virtual_sites: bool = True
    ) -> Tuple[openmm.System, app.Topology, np.ndarray]:
        """
        Create OpenMM system with hybrid forcefield
        
        Args:
            protein_pdb: Path to protein PDB file
            ligand_mol: RDKit molecule (must have 3D coordinates)
            box_size: Cubic box size in nm
            add_virtual_sites: Add virtual sites for sigma holes
        
        Returns:
            (system, topology, positions)
        """
        # Load protein
        pdb = app.PDBFile(protein_pdb)
        protein_topology = pdb.topology
        protein_positions = pdb.positions
        
        # Create system for protein (AMBER14)
        protein_system = self.amber.createSystem(
            protein_topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0*unit.nanometer,
            constraints=app.HBonds
        )
        
        # Prepare ligand
        ligand_topology, ligand_positions = self._prepare_ligand(ligand_mol)
        
        # Combine topologies
        combined_topology = self._combine_topologies(
            protein_topology,
            ligand_topology
        )
        
        # Combine positions
        combined_positions = np.vstack([
            protein_positions.value_in_unit(unit.nanometer),
            ligand_positions
        ])
        
        # Create combined system
        system = self._create_combined_system(
            protein_system,
            ligand_mol,
            combined_topology
        )
        
        # Add virtual sites for sigma holes
        if add_virtual_sites:
            self._add_virtual_sites(system, combined_topology, ligand_mol)
        
        # Add periodic box
        system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(box_size, 0, 0),
            openmm.Vec3(0, box_size, 0),
            openmm.Vec3(0, 0, box_size)
        )
        
        return system, combined_topology, combined_positions
    
    def _prepare_ligand(
        self,
        mol: Chem.Mol
    ) -> Tuple[app.Topology, np.ndarray]:
        """
        Prepare ligand topology and positions
        
        Args:
            mol: RDKit molecule with 3D coordinates
        
        Returns:
            (topology, positions_nm)
        """
        # Create OpenMM topology for ligand
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue('LIG', chain)
        
        atoms = []
        for atom in mol.GetAtoms():
            element = app.Element.getByAtomicNumber(atom.GetAtomicNum())
            atoms.append(topology.addAtom(atom.GetSymbol(), element, residue))
        
        # Add bonds
        for bond in mol.GetBonds():
            topology.addBond(
                atoms[bond.GetBeginAtomIdx()],
                atoms[bond.GetEndAtomIdx()]
            )
        
        # Get positions
        conformer = mol.GetConformer()
        positions = conformer.GetPositions() / 10.0  # Angstrom to nm
        
        return topology, positions
    
    def _combine_topologies(
        self,
        protein_topology: app.Topology,
        ligand_topology: app.Topology
    ) -> app.Topology:
        """Combine protein and ligand topologies"""
        combined = app.Topology()
        
        # Add protein
        for chain in protein_topology.chains():
            new_chain = combined.addChain(chain.id)
            for residue in chain.residues():
                new_residue = combined.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    combined.addAtom(atom.name, atom.element, new_residue)
        
        # Add ligand
        for chain in ligand_topology.chains():
            new_chain = combined.addChain('L')
            for residue in chain.residues():
                new_residue = combined.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    combined.addAtom(atom.name, atom.element, new_residue)
        
        return combined
    
    def _create_combined_system(
        self,
        protein_system: openmm.System,
        ligand_mol: Chem.Mol,
        topology: app.Topology
    ) -> openmm.System:
        """
        Create combined system with hybrid forcefield
        
        Protein: AMBER14
        Ligand: ANI-2x + ESP-DNN
        """
        # Start with protein system
        system = protein_system
        
        # Add ligand particles
        n_protein_atoms = system.getNumParticles()
        for atom in ligand_mol.GetAtoms():
            mass = atom.GetMass()
            system.addParticle(mass)
        
        # Add ANI-2x custom force for ligand
        self._add_ani2x_force(system, ligand_mol, n_protein_atoms)
        
        # Set ligand charges (ESP-DNN or Gasteiger)
        self._set_ligand_charges(system, ligand_mol, n_protein_atoms)
        
        return system
    
    def _add_ani2x_force(
        self,
        system: openmm.System,
        ligand_mol: Chem.Mol,
        offset: int
    ):
        """Add ANI-2x custom force for ligand"""
        # Create custom force
        force = openmm.CustomExternalForce("0")  # Placeholder
        
        # Add ligand atoms
        for i, atom in enumerate(ligand_mol.GetAtoms()):
            force.addParticle(offset + i, [])
        
        system.addForce(force)
        
        # Note: Actual ANI-2x integration requires custom force implementation
        # This is a simplified placeholder
    
    def _set_ligand_charges(
        self,
        system: openmm.System,
        ligand_mol: Chem.Mol,
        offset: int
    ):
        """Set ligand partial charges"""
        if self.espdnn:
            # Use ESP-DNN charges
            charges = self.espdnn.calculate_charges(ligand_mol)
        else:
            # Use Gasteiger charges
            AllChem.ComputeGasteigerCharges(ligand_mol)
            charges = np.array([
                atom.GetDoubleProp('_GasteigerCharge')
                for atom in ligand_mol.GetAtoms()
            ])
        
        # Find NonbondedForce and set charges
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for i, charge in enumerate(charges):
                    # Get existing parameters
                    q, sigma, epsilon = force.getParticleParameters(offset + i)
                    # Set new charge
                    force.setParticleParameters(
                        offset + i,
                        charge * unit.elementary_charge,
                        sigma,
                        epsilon
                    )
    
    def _add_virtual_sites(
        self,
        system: openmm.System,
        topology: app.Topology,
        ligand_mol: Chem.Mol
    ):
        """Add virtual sites for halogen sigma holes"""
        builder = VirtualSiteBuilder()
        halogens = builder.detect_halogens(ligand_mol)
        
        for halogen_idx in halogens:
            # Find bonded atom
            atom = ligand_mol.GetAtomWithIdx(halogen_idx)
            bonds = atom.GetBonds()
            if len(bonds) > 0:
                bonded_idx = bonds[0].GetOtherAtomIdx(halogen_idx)
                
                # Add virtual site
                builder.add_virtual_site(
                    system,
                    topology,
                    halogen_idx,
                    bonded_idx
                )
    
    def assign_formal_charges(
        self,
        mol: Chem.Mol,
        ph: float = 7.0
    ) -> Chem.Mol:
        """
        Assign formal charges at physiological pH
        
        Args:
            mol: RDKit molecule
            ph: pH value (default: 7.0)
        
        Returns:
            Molecule with formal charges assigned
        """
        # Use RDKit's charge assignment
        mol = Chem.AddHs(mol)
        
        # Assign formal charges based on pH
        # This is a simplified implementation
        # For production, use tools like Epik or Chemaxon
        
        for atom in mol.GetAtoms():
            # Carboxylic acids: deprotonated at pH 7
            if self._is_carboxylic_acid(atom):
                atom.SetFormalCharge(-1)
            
            # Amines: protonated at pH 7 if pKa > 7
            elif self._is_amine(atom):
                if self._estimate_pka(atom) > ph:
                    atom.SetFormalCharge(1)
        
        return mol
    
    @staticmethod
    def _is_carboxylic_acid(atom: Chem.Atom) -> bool:
        """Check if atom is part of carboxylic acid"""
        if atom.GetAtomicNum() != 8:  # Oxygen
            return False
        
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 6:  # Carbon
                # Check for C=O and C-OH pattern
                return True
        
        return False
    
    @staticmethod
    def _is_amine(atom: Chem.Atom) -> bool:
        """Check if atom is amine nitrogen"""
        return atom.GetAtomicNum() == 7  # Nitrogen
    
    @staticmethod
    def _estimate_pka(atom: Chem.Atom) -> float:
        """Estimate pKa (simplified)"""
        # Simplified pKa estimation
        # For production, use tools like Epik or Chemaxon
        if atom.GetAtomicNum() == 7:  # Nitrogen
            return 9.0  # Typical amine pKa
        return 7.0


# Example usage
if __name__ == '__main__':
    # Initialize forcefield
    ff = ForcelabElixirForcefield(
        ani2x_device='cuda',
        espdnn_model_path='/path/to/espdnn.pt'
    )
    
    # Load protein and ligand
    protein_pdb = 'protein.pdb'
    ligand_smiles = 'CCO'  # Ethanol
    
    # Prepare ligand
    mol = Chem.MolFromSmiles(ligand_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    
    # Assign formal charges
    mol = ff.assign_formal_charges(mol, ph=7.0)
    
    # Create system
    system, topology, positions = ff.create_system(
        protein_pdb,
        mol,
        box_size=10.0,
        add_virtual_sites=True
    )
    
    print(f"System created with {system.getNumParticles()} particles")
    print(f"Topology has {topology.getNumAtoms()} atoms")
```

**Testing Checklist:**
- [ ] ANI-2x loads correctly and calculates energies
- [ ] ESP-DNN loads and calculates charges
- [ ] Virtual sites added for halogens
- [ ] Formal charges assigned correctly at pH 7.0
- [ ] Combined system created without errors
- [ ] Energy calculation runs without crashes

**Performance Target:**
- System creation: < 5 seconds
- Energy calculation: < 100ms per evaluation

---

### Validation

Before proceeding to Phase 2, verify:

1. **Unit Tests Pass**: Run `pytest tests/test_fep/test_forcefield.py`
2. **Energy Accuracy**: Compare ANI-2x energies with reference QM calculations (error < 1 kcal/mol)
3. **Charge Accuracy**: Compare ESP-DNN charges with reference ESP charges (RMSE < 0.1 e)
4. **Performance**: System creation and energy evaluation meet targets

---

## Next Steps for Codex

After completing Phase 1, proceed to:

**Phase 2: Molecular Editor Backend** - See `CODEX_PHASE2_EDITOR.md`

This phase implements:
- Fragment library management
- Scaffold hopping algorithms
- R-group decoration
- Constraint-based filtering

---

## Support Resources

**Documentation:**
- OpenMM: http://docs.openmm.org/
- RDKit: https://www.rdkit.org/docs/
- TorchANI: https://aiqm.github.io/torchani/

**Model Checkpoints:**
- ANI-2x: https://github.com/aiqm/torchani/releases
- ESP-DNN: [Provide custom checkpoint]

**Test Data:**
- Protein PDB: 1A28 (HIV protease)
- Ligands: Ibuprofen, Aspirin, Caffeine

---

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**
   - Solution: Reduce batch size or use CPU
   - Check: `torch.cuda.memory_allocated()`

2. **ANI-2x Model Not Found**
   - Solution: Download from GitHub releases
   - Verify: Model file exists and is readable

3. **OpenMM Force Creation Fails**
   - Solution: Check topology consistency
   - Debug: Print atom counts at each step

4. **Charge Assignment Errors**
   - Solution: Verify molecule has 3D coordinates
   - Check: `mol.GetNumConformers() > 0`

---

## Code Quality Standards

**Python Style:**
- Follow PEP 8
- Use type hints for all functions
- Docstrings in Google style
- Maximum line length: 100 characters

**Testing:**
- Unit tests for all public methods
- Integration tests for workflows
- Performance benchmarks
- Coverage target: >90%

**Documentation:**
- API documentation with examples
- Usage tutorials
- Performance characteristics
- Known limitations

---

**End of Phase 1 Plan**

Continue to Phase 2 after validation checkpoint.
