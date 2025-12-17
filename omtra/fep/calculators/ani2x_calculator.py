"""
ANI-2x Energy Calculator for FEP Calculations

This module provides quantum-accurate energy calculations using the ANI-2x neural network potential.
ANI-2x achieves chemical accuracy (~1 kcal/mol) for organic molecules containing H, C, N, O, F, S, Cl.

References:
    Devereux et al. "Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens"
    J. Chem. Theory Comput. 2020, 16, 7, 4192-4202
"""

import torch
import torchani
import numpy as np
from typing import Optional, Tuple, Dict, List
from rdkit import Chem
from rdkit.Chem import AllChem


class ANI2xCalculator:
    """
    Quantum-accurate energy calculator using ANI-2x neural network potential.
    
    ANI-2x provides DFT-level accuracy for molecular energies with GPU acceleration,
    making it suitable for fast FEP estimation and intermediate FEP calculations.
    
    Attributes:
        model: TorchANI ANI-2x model
        device: torch.device for computation (CPU or CUDA)
        supported_elements: Set of atomic numbers supported by ANI-2x
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize ANI-2x calculator.
        
        Args:
            device: Device for computation ('cpu', 'cuda', or None for auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load ANI-2x model
        self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # ANI-2x supports H, C, N, O, F, S, Cl, Br, I
        self.supported_elements = {1, 6, 7, 8, 9, 16, 17, 35, 53}
        
        # Energy unit conversion (Hartree to kcal/mol)
        self.hartree_to_kcal = 627.509474
        
    def is_supported(self, mol: Chem.Mol) -> Tuple[bool, List[int]]:
        """
        Check if molecule contains only ANI-2x supported elements.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tuple of (is_supported, unsupported_atoms)
        """
        unsupported = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num not in self.supported_elements:
                unsupported.append(atomic_num)
        
        return len(unsupported) == 0, unsupported
    
    def mol_to_ani_input(self, mol: Chem.Mol, conf_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert RDKit molecule to ANI-2x input format.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            conf_id: Conformer ID to use
            
        Returns:
            Tuple of (species, coordinates) tensors
        """
        # Get atomic numbers
        species = torch.tensor([[atom.GetAtomicNum() for atom in mol.GetAtoms()]], 
                              dtype=torch.long, device=self.device)
        
        # Get 3D coordinates
        conf = mol.GetConformer(conf_id)
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        
        coordinates = torch.tensor([coords], dtype=torch.float32, device=self.device)
        
        return species, coordinates
    
    def calculate_energy(self, mol: Chem.Mol, conf_id: int = 0, 
                        return_forces: bool = False) -> Dict[str, float]:
        """
        Calculate molecular energy using ANI-2x.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            conf_id: Conformer ID to use
            return_forces: Whether to compute forces (gradients)
            
        Returns:
            Dictionary with 'energy' in kcal/mol and optionally 'forces'
            
        Raises:
            ValueError: If molecule contains unsupported elements
        """
        # Check if molecule is supported
        is_supported, unsupported = self.is_supported(mol)
        if not is_supported:
            raise ValueError(f"Molecule contains unsupported elements: {unsupported}")
        
        # Convert to ANI input format
        species, coordinates = self.mol_to_ani_input(mol, conf_id)
        
        # Enable gradient computation if forces are needed
        if return_forces:
            coordinates.requires_grad_(True)
        
        # Calculate energy
        with torch.set_grad_enabled(return_forces):
            energy = self.model((species, coordinates)).energies
        
        # Convert to kcal/mol
        energy_kcal = energy.item() * self.hartree_to_kcal
        
        result = {'energy': energy_kcal}
        
        # Calculate forces if requested
        if return_forces:
            forces = -torch.autograd.grad(energy.sum(), coordinates)[0]
            result['forces'] = forces.cpu().detach().numpy()
        
        return result
    
    def calculate_energy_batch(self, mols: List[Chem.Mol], 
                              conf_ids: Optional[List[int]] = None) -> List[Dict[str, float]]:
        """
        Calculate energies for multiple molecules in batch (GPU-accelerated).
        
        Args:
            mols: List of RDKit molecules with 3D coordinates
            conf_ids: List of conformer IDs (defaults to 0 for all)
            
        Returns:
            List of dictionaries with energies in kcal/mol
        """
        if conf_ids is None:
            conf_ids = [0] * len(mols)
        
        results = []
        
        # Process in batches for GPU efficiency
        batch_size = 32
        for i in range(0, len(mols), batch_size):
            batch_mols = mols[i:i+batch_size]
            batch_conf_ids = conf_ids[i:i+batch_size]
            
            # Prepare batch data
            species_list = []
            coords_list = []
            
            for mol, conf_id in zip(batch_mols, batch_conf_ids):
                species, coords = self.mol_to_ani_input(mol, conf_id)
                species_list.append(species.squeeze(0))
                coords_list.append(coords.squeeze(0))
            
            # Pad sequences to same length
            max_atoms = max(s.shape[0] for s in species_list)
            
            padded_species = []
            padded_coords = []
            
            for species, coords in zip(species_list, coords_list):
                n_atoms = species.shape[0]
                if n_atoms < max_atoms:
                    # Pad with zeros (will be masked)
                    species_pad = torch.zeros(max_atoms, dtype=torch.long, device=self.device)
                    species_pad[:n_atoms] = species
                    
                    coords_pad = torch.zeros((max_atoms, 3), dtype=torch.float32, device=self.device)
                    coords_pad[:n_atoms] = coords
                    
                    padded_species.append(species_pad)
                    padded_coords.append(coords_pad)
                else:
                    padded_species.append(species)
                    padded_coords.append(coords)
            
            # Stack into batch tensors
            batch_species = torch.stack(padded_species)
            batch_coords = torch.stack(padded_coords)
            
            # Calculate energies
            with torch.no_grad():
                batch_energies = self.model((batch_species, batch_coords)).energies
            
            # Convert to kcal/mol and store
            for energy in batch_energies:
                results.append({'energy': energy.item() * self.hartree_to_kcal})
        
        return results
    
    def calculate_interaction_energy(self, complex_mol: Chem.Mol, 
                                    ligand_mol: Chem.Mol,
                                    protein_mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate protein-ligand interaction energy.
        
        E_interaction = E_complex - (E_ligand + E_protein)
        
        Args:
            complex_mol: Protein-ligand complex
            ligand_mol: Ligand alone
            protein_mol: Protein alone
            
        Returns:
            Dictionary with interaction energy in kcal/mol
        """
        # Calculate individual energies
        e_complex = self.calculate_energy(complex_mol)['energy']
        e_ligand = self.calculate_energy(ligand_mol)['energy']
        e_protein = self.calculate_energy(protein_mol)['energy']
        
        # Interaction energy
        e_interaction = e_complex - (e_ligand + e_protein)
        
        return {
            'interaction_energy': e_interaction,
            'complex_energy': e_complex,
            'ligand_energy': e_ligand,
            'protein_energy': e_protein
        }
    
    def optimize_geometry(self, mol: Chem.Mol, conf_id: int = 0, 
                         max_steps: int = 500, 
                         convergence: float = 1e-4) -> Tuple[Chem.Mol, Dict]:
        """
        Optimize molecular geometry using ANI-2x forces.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            conf_id: Conformer ID to optimize
            max_steps: Maximum optimization steps
            convergence: Energy convergence criterion (kcal/mol)
            
        Returns:
            Tuple of (optimized_mol, optimization_info)
        """
        # Clone molecule to avoid modifying original
        opt_mol = Chem.Mol(mol)
        
        # Get initial energy
        prev_energy = self.calculate_energy(opt_mol, conf_id)['energy']
        
        # Simple gradient descent optimization
        learning_rate = 0.01
        
        for step in range(max_steps):
            # Calculate energy and forces
            result = self.calculate_energy(opt_mol, conf_id, return_forces=True)
            energy = result['energy']
            forces = result['forces'].squeeze(0)  # Remove batch dimension
            
            # Check convergence
            if abs(energy - prev_energy) < convergence:
                return opt_mol, {
                    'converged': True,
                    'steps': step,
                    'final_energy': energy,
                    'energy_change': energy - prev_energy
                }
            
            # Update coordinates
            conf = opt_mol.GetConformer(conf_id)
            for i in range(opt_mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                new_pos = [
                    pos.x + learning_rate * forces[i, 0],
                    pos.y + learning_rate * forces[i, 1],
                    pos.z + learning_rate * forces[i, 2]
                ]
                conf.SetAtomPosition(i, new_pos)
            
            prev_energy = energy
        
        # Did not converge
        return opt_mol, {
            'converged': False,
            'steps': max_steps,
            'final_energy': energy,
            'energy_change': energy - prev_energy
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the ANI-2x model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': 'ANI-2x',
            'device': str(self.device),
            'supported_elements': sorted(list(self.supported_elements)),
            'accuracy': '~1 kcal/mol (DFT-level)',
            'reference': 'Devereux et al. JCTC 2020, 16, 7, 4192-4202'
        }


# Example usage
if __name__ == "__main__":
    # Initialize calculator
    calc = ANI2xCalculator()
    
    print(f"ANI-2x Calculator initialized on {calc.device}")
    print(f"Model info: {calc.get_model_info()}")
    
    # Create a simple molecule (ethanol)
    mol = Chem.MolFromSmiles('CCO')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeM olecule(mol)
    
    # Calculate energy
    result = calc.calculate_energy(mol)
    print(f"\nEthanol energy: {result['energy']:.2f} kcal/mol")
    
    # Check if molecule is supported
    is_supported, unsupported = calc.is_supported(mol)
    print(f"Molecule supported: {is_supported}")
