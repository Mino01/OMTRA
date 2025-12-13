"""
Example: Using Convergence Monitoring with OMTRA

This example demonstrates how to use the convergence monitoring module
to assess the quality and reliability of OMTRA molecular generation.

Author: Manus AI
Date: December 11, 2025
"""

import numpy as np
from pathlib import Path
import sys

# Add OMTRA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omtra.convergence.monitor import ConvergenceMonitor, print_convergence_report


def example_basic_monitoring():
    """Basic convergence monitoring example"""
    print("="*60)
    print("Example 1: Basic Convergence Monitoring")
    print("="*60)
    
    # Simulated generated molecules (SMILES)
    molecules = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "CCCO",  # Propanol
        "CC(C)(C)O",  # tert-Butanol
        "CCCCO",  # Butanol
        "CC(C)CO",  # Isobutanol
        "CCC(C)O",  # sec-Butanol
        "CCCCCO",  # Pentanol
        "CC(C)CCO",  # Isopentanol
        "CCC(C)(C)O",  # tert-Pentanol
    ]
    
    # Simulated energy values
    energies = np.random.normal(-50, 10, len(molecules))
    
    # Initialize monitor
    monitor = ConvergenceMonitor()
    
    # Run monitoring
    metrics = monitor.monitor(
        molecules=molecules,
        energies=energies,
        task_type='denovo_ligand_condensed'
    )
    
    # Print report
    print_convergence_report(metrics)
    
    # Save metrics
    output_path = Path('/tmp/convergence_metrics.json')
    metrics.save(output_path)
    print(f"Metrics saved to: {output_path}")


def example_with_trajectories():
    """Example with forward/backward trajectories"""
    print("\n" + "="*60)
    print("Example 2: Monitoring with Forward/Backward Trajectories")
    print("="*60)
    
    # Simulated molecules
    molecules = [
        "c1ccccc1",  # Benzene
        "c1ccc(O)cc1",  # Phenol
        "c1ccc(N)cc1",  # Aniline
        "c1ccc(C)cc1",  # Toluene
        "c1ccc(F)cc1",  # Fluorobenzene
        "c1ccc(Cl)cc1",  # Chlorobenzene
        "c1ccc(Br)cc1",  # Bromobenzene
        "c1ccc(I)cc1",  # Iodobenzene
        "c1ccc(C(F)(F)F)cc1",  # Trifluorotoluene
        "c1ccc(OC)cc1",  # Anisole
    ]
    
    # Simulated forward and backward energies
    forward_energies = np.random.normal(-45, 8, 100)
    backward_energies = forward_energies + np.random.normal(0, 1, 100)  # Small hysteresis
    
    # Simulated weights (should be relatively uniform for good sampling)
    weights = np.random.dirichlet(np.ones(len(molecules)))
    
    # Initialize monitor
    monitor = ConvergenceMonitor()
    
    # Run monitoring
    metrics = monitor.monitor(
        molecules=molecules,
        forward_energies=forward_energies,
        backward_energies=backward_energies,
        weights=weights,
        task_type='protein_conditioned_ligand'
    )
    
    # Print report
    print_convergence_report(metrics)


def example_with_reference():
    """Example with reference molecules for novelty assessment"""
    print("\n" + "="*60)
    print("Example 3: Novelty Assessment with Reference Set")
    print("="*60)
    
    # Generated molecules
    generated = [
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)NCC(COc1ccccc1)O",  # Propranolol
        "CN(C)CCOC(c1ccccc1)c1ccccc1",  # Diphenhydramine
    ]
    
    # Reference molecules (training set)
    reference = [
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen (same as generated)
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin (same as generated)
        "COc1ccc2nc(S(N)(=O)=O)sc2c1",  # Different molecule
        "Cc1ccc(C(=O)O)cc1",  # Different molecule
        "CCOc1ccc(CC(=O)O)cc1",  # Different molecule
    ]
    
    # Initialize monitor
    monitor = ConvergenceMonitor()
    
    # Run monitoring with novelty assessment
    metrics = monitor.monitor(
        molecules=generated,
        reference_molecules=reference,
        task_type='denovo_ligand_condensed'
    )
    
    # Print report
    print_convergence_report(metrics)
    
    print(f"\nNovelty Analysis:")
    print(f"  Generated: {len(generated)} molecules")
    print(f"  Reference: {len(reference)} molecules")
    print(f"  Novel molecules: {int(metrics.novelty_score * len(generated))}")


def example_poor_convergence():
    """Example demonstrating poor convergence"""
    print("\n" + "="*60)
    print("Example 4: Poor Convergence Detection")
    print("="*60)
    
    # Simulated molecules with low diversity (all similar)
    molecules = [
        "CCO",
        "CCCO",
        "CCCCO",
        "CCCCCO",
        "CCCCCCO",
    ]
    
    # High hysteresis (inconsistent trajectories)
    forward_energies = np.random.normal(-50, 5, 50)
    backward_energies = np.random.normal(-40, 5, 50)  # Large difference
    
    # Poor sampling (one sample dominates)
    weights = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
    
    # Initialize monitor
    monitor = ConvergenceMonitor()
    
    # Run monitoring
    metrics = monitor.monitor(
        molecules=molecules,
        forward_energies=forward_energies,
        backward_energies=backward_energies,
        weights=weights,
        task_type='denovo_ligand_condensed'
    )
    
    # Print report
    print_convergence_report(metrics)
    
    print("\nThis example demonstrates multiple convergence issues:")
    print("  - High hysteresis (inconsistent generation)")
    print("  - Low diversity (similar molecules)")
    print("  - Poor sampling (single sample dominates)")
    print("\nThe recommendations suggest how to improve convergence.")


def example_custom_thresholds():
    """Example with custom convergence thresholds"""
    print("\n" + "="*60)
    print("Example 5: Custom Convergence Thresholds")
    print("="*60)
    
    # Molecules
    molecules = [
        "c1ccccc1C(=O)O",  # Benzoic acid
        "c1ccc(cc1)C(=O)O",  # Benzoic acid (different SMILES)
        "Cc1ccccc1C(=O)O",  # Toluic acid
        "Cc1ccc(C)cc1C(=O)O",  # Dimethylbenzoic acid
        "c1ccc(O)c(C(=O)O)c1",  # Salicylic acid
    ]
    
    # Custom thresholds (more strict)
    custom_thresholds = {
        'hysteresis': 1.0,  # More strict (default: 2.0)
        'bhattacharyya': 0.02,  # More strict (default: 0.03)
        'max_weight': 0.03,  # More strict (default: 0.05)
        'diversity': 0.8,  # More strict (default: 0.7)
        'validity_rate': 0.98,  # More strict (default: 0.95)
        'novelty_score': 0.6,  # More strict (default: 0.5)
    }
    
    # Initialize monitor with custom thresholds
    monitor = ConvergenceMonitor(thresholds=custom_thresholds)
    
    # Run monitoring
    metrics = monitor.monitor(
        molecules=molecules,
        task_type='denovo_ligand_condensed'
    )
    
    # Print report
    print_convergence_report(metrics)
    
    print("\nCustom thresholds applied:")
    for metric, threshold in custom_thresholds.items():
        print(f"  {metric}: {threshold}")


if __name__ == '__main__':
    # Run all examples
    example_basic_monitoring()
    example_with_trajectories()
    example_with_reference()
    example_poor_convergence()
    example_custom_thresholds()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
