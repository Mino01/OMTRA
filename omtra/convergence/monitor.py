"""
Convergence Monitoring for OMTRA Molecular Generation

This module implements convergence monitoring based on Free Energy Perturbation (FEP)
best practices from Martin Olsson's research, adapted for molecular generation tasks.

Key metrics:
- Forward-Backward Hysteresis: Measures consistency of generation process
- Bhattacharyya Coefficient: Measures overlap between distributions
- Maximum Weight: Detects poor sampling
- Diversity Score: Measures molecular diversity
- Statistical Significance: Provides confidence intervals

Author: Manus AI
Date: December 11, 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ConvergenceMetrics:
    """Container for convergence metrics"""
    hysteresis: float
    bhattacharyya: float
    max_weight: float
    diversity: float
    validity_rate: float
    novelty_score: float
    converged: bool
    failed_metrics: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'hysteresis': float(self.hysteresis),
            'bhattacharyya': float(self.bhattacharyya),
            'max_weight': float(self.max_weight),
            'diversity': float(self.diversity),
            'validity_rate': float(self.validity_rate),
            'novelty_score': float(self.novelty_score),
            'converged': bool(self.converged),
            'failed_metrics': self.failed_metrics,
            'recommendations': self.recommendations,
        }
    
    def save(self, path: Path):
        """Save metrics to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ConvergenceMonitor:
    """
    Monitor convergence of molecular generation tasks
    
    Implements convergence metrics adapted from FEP methodology:
    - Hysteresis: Measures forward-backward consistency
    - Bhattacharyya: Measures distribution overlap
    - Max Weight: Detects poor sampling
    - Diversity: Measures molecular diversity
    
    Thresholds are based on empirical studies and can be adjusted.
    """
    
    DEFAULT_THRESHOLDS = {
        'hysteresis': 2.0,  # kJ/mol equivalent
        'bhattacharyya': 0.03,  # Overlap threshold
        'max_weight': 0.05,  # Maximum single weight
        'diversity': 0.7,  # Minimum diversity score
        'validity_rate': 0.95,  # Minimum validity rate
        'novelty_score': 0.5,  # Minimum novelty
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize convergence monitor
        
        Args:
            thresholds: Custom thresholds for convergence metrics
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
    
    def calculate_hysteresis(
        self,
        forward_energies: np.ndarray,
        backward_energies: np.ndarray
    ) -> float:
        """
        Calculate forward-backward hysteresis
        
        Measures the consistency of the generation process by comparing
        forward and backward trajectories. Lower values indicate better
        convergence.
        
        Args:
            forward_energies: Energy values from forward trajectory
            backward_energies: Energy values from backward trajectory
            
        Returns:
            Hysteresis value (lower is better)
        """
        if len(forward_energies) == 0 or len(backward_energies) == 0:
            return float('inf')
        
        # Calculate mean absolute difference
        hysteresis = np.abs(np.mean(forward_energies) - np.mean(backward_energies))
        
        # Add standard deviation penalty for inconsistency
        std_penalty = np.abs(np.std(forward_energies) - np.std(backward_energies))
        
        return float(hysteresis + 0.5 * std_penalty)
    
    def calculate_bhattacharyya(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray,
        bins: int = 50
    ) -> float:
        """
        Calculate Bhattacharyya coefficient
        
        Measures the overlap between two probability distributions.
        Lower values indicate better overlap and convergence.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            bins: Number of bins for histogram
            
        Returns:
            Bhattacharyya coefficient (lower is better, 0 = perfect overlap)
        """
        if len(dist1) == 0 or len(dist2) == 0:
            return 1.0
        
        # Create histograms with same bins
        min_val = min(np.min(dist1), np.min(dist2))
        max_val = max(np.max(dist1), np.max(dist2))
        
        hist1, _ = np.histogram(dist1, bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val), density=True)
        
        # Normalize
        hist1 = hist1 / (np.sum(hist1) + 1e-10)
        hist2 = hist2 / (np.sum(hist2) + 1e-10)
        
        # Calculate Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist1 * hist2))
        
        # Return distance (1 - coefficient)
        return float(1.0 - bc)
    
    def calculate_max_weight(
        self,
        weights: np.ndarray
    ) -> float:
        """
        Calculate maximum weight in ensemble
        
        Detects poor sampling by identifying if a single sample dominates
        the ensemble. Lower values indicate better sampling.
        
        Args:
            weights: Sample weights (e.g., probabilities, energies)
            
        Returns:
            Maximum normalized weight (lower is better)
        """
        if len(weights) == 0:
            return 1.0
        
        # Normalize weights
        normalized = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-10)
        
        return float(np.max(normalized))
    
    def calculate_diversity(
        self,
        molecules: List[str],
        fingerprint_type: str = 'morgan'
    ) -> float:
        """
        Calculate molecular diversity using Tanimoto similarity
        
        Measures the diversity of generated molecules. Higher values
        indicate more diverse generation.
        
        Args:
            molecules: List of SMILES strings
            fingerprint_type: Type of fingerprint ('morgan', 'maccs', 'topological')
            
        Returns:
            Diversity score (higher is better, 0-1 range)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, MACCSkeys
            from rdkit.DataStructs import TanimotoSimilarity
        except ImportError:
            print("Warning: RDKit not available, skipping diversity calculation")
            return 0.0
        
        if len(molecules) < 2:
            return 0.0
        
        # Generate fingerprints
        fps = []
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            if fingerprint_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            elif fingerprint_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fingerprint_type == 'topological':
                fp = Chem.RDKFingerprint(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")
            
            fps.append(fp)
        
        if len(fps) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        # Diversity = 1 - mean similarity
        if similarities:
            diversity = 1.0 - np.mean(similarities)
            return float(np.clip(diversity, 0.0, 1.0))
        
        return 0.0
    
    def calculate_validity_rate(
        self,
        molecules: List[str]
    ) -> float:
        """
        Calculate the rate of chemically valid molecules
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Validity rate (0-1 range)
        """
        try:
            from rdkit import Chem
        except ImportError:
            print("Warning: RDKit not available, skipping validity calculation")
            return 0.0
        
        if len(molecules) == 0:
            return 0.0
        
        valid_count = sum(1 for smiles in molecules if Chem.MolFromSmiles(smiles) is not None)
        return float(valid_count / len(molecules))
    
    def calculate_novelty(
        self,
        generated_molecules: List[str],
        reference_molecules: List[str],
        threshold: float = 0.4
    ) -> float:
        """
        Calculate novelty score compared to reference set
        
        Args:
            generated_molecules: Generated SMILES
            reference_molecules: Reference SMILES (e.g., training set)
            threshold: Similarity threshold for novelty
            
        Returns:
            Novelty score (fraction of novel molecules)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.DataStructs import TanimotoSimilarity
        except ImportError:
            print("Warning: RDKit not available, skipping novelty calculation")
            return 0.0
        
        if len(generated_molecules) == 0 or len(reference_molecules) == 0:
            return 0.0
        
        # Generate fingerprints for reference
        ref_fps = []
        for smiles in reference_molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                ref_fps.append(fp)
        
        if len(ref_fps) == 0:
            return 0.0
        
        # Check novelty of generated molecules
        novel_count = 0
        for smiles in generated_molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            
            # Calculate max similarity to reference set
            max_sim = max(TanimotoSimilarity(fp, ref_fp) for ref_fp in ref_fps)
            
            # Novel if max similarity below threshold
            if max_sim < threshold:
                novel_count += 1
        
        return float(novel_count / len(generated_molecules))
    
    def assess_convergence(
        self,
        metrics: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Assess overall convergence based on all metrics
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Tuple of (converged, failed_metrics)
        """
        failed = []
        
        for metric, value in metrics.items():
            if metric not in self.thresholds:
                continue
            
            threshold = self.thresholds[metric]
            
            # Different metrics have different pass conditions
            if metric in ['hysteresis', 'bhattacharyya', 'max_weight']:
                # Lower is better
                if value > threshold:
                    failed.append(metric)
            elif metric in ['diversity', 'validity_rate', 'novelty_score']:
                # Higher is better
                if value < threshold:
                    failed.append(metric)
        
        converged = len(failed) == 0
        return converged, failed
    
    def generate_recommendations(
        self,
        failed_metrics: List[str],
        task_type: str = 'denovo'
    ) -> List[str]:
        """
        Generate actionable recommendations for failed metrics
        
        Args:
            failed_metrics: List of metrics that failed thresholds
            task_type: Type of OMTRA task
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if 'hysteresis' in failed_metrics:
            recommendations.append(
                "High hysteresis detected. Increase n_timesteps (e.g., from 250 to 500) "
                "for more stable generation trajectory."
            )
        
        if 'bhattacharyya' in failed_metrics:
            recommendations.append(
                "Poor distribution overlap. Run multiple independent generations "
                "(increase n_replicates) and use ensemble averaging."
            )
        
        if 'max_weight' in failed_metrics:
            recommendations.append(
                "Poor sampling detected (single sample dominates). "
                "Enable stochastic_sampling or increase noise_scaler for more exploration."
            )
        
        if 'diversity' in failed_metrics:
            recommendations.append(
                "Low molecular diversity. Increase n_samples or temperature parameter. "
                "Consider using different random seeds for multiple runs."
            )
        
        if 'validity_rate' in failed_metrics:
            recommendations.append(
                "Low chemical validity rate. Check model checkpoint quality. "
                "Consider post-processing with validity filters or constraint satisfaction."
            )
        
        if 'novelty_score' in failed_metrics:
            recommendations.append(
                "Low novelty score (too similar to training set). "
                "Adjust sampling parameters or use conditional generation with novel constraints."
            )
        
        # Task-specific recommendations
        if task_type == 'protein_conditioned_ligand':
            recommendations.append(
                "For protein-conditioned generation: verify binding site coordinates "
                "and ensure sufficient box_size for ligand exploration."
            )
        elif task_type == 'docking':
            recommendations.append(
                "For docking tasks: increase n_poses and verify protein preparation "
                "(protonation, missing residues, etc.)."
            )
        
        return recommendations
    
    def monitor(
        self,
        molecules: List[str],
        energies: Optional[np.ndarray] = None,
        forward_energies: Optional[np.ndarray] = None,
        backward_energies: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        reference_molecules: Optional[List[str]] = None,
        task_type: str = 'denovo'
    ) -> ConvergenceMetrics:
        """
        Comprehensive convergence monitoring
        
        Args:
            molecules: Generated SMILES strings
            energies: Energy values for molecules
            forward_energies: Forward trajectory energies
            backward_energies: Backward trajectory energies
            weights: Sample weights
            reference_molecules: Reference molecules for novelty
            task_type: OMTRA task type
            
        Returns:
            ConvergenceMetrics object with all metrics and recommendations
        """
        metrics = {}
        
        # Calculate hysteresis if trajectories provided
        if forward_energies is not None and backward_energies is not None:
            metrics['hysteresis'] = self.calculate_hysteresis(
                forward_energies, backward_energies
            )
        else:
            metrics['hysteresis'] = 0.0
        
        # Calculate Bhattacharyya if we have energies
        if energies is not None and len(energies) > 1:
            # Split into two halves for comparison
            mid = len(energies) // 2
            metrics['bhattacharyya'] = self.calculate_bhattacharyya(
                energies[:mid], energies[mid:]
            )
        else:
            metrics['bhattacharyya'] = 0.0
        
        # Calculate max weight if provided
        if weights is not None:
            metrics['max_weight'] = self.calculate_max_weight(weights)
        else:
            # Use uniform weights
            metrics['max_weight'] = 1.0 / max(len(molecules), 1)
        
        # Calculate diversity
        metrics['diversity'] = self.calculate_diversity(molecules)
        
        # Calculate validity rate
        metrics['validity_rate'] = self.calculate_validity_rate(molecules)
        
        # Calculate novelty if reference provided
        if reference_molecules:
            metrics['novelty_score'] = self.calculate_novelty(
                molecules, reference_molecules
            )
        else:
            metrics['novelty_score'] = 1.0  # Assume novel if no reference
        
        # Assess convergence
        converged, failed_metrics = self.assess_convergence(metrics)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(failed_metrics, task_type)
        
        return ConvergenceMetrics(
            hysteresis=metrics['hysteresis'],
            bhattacharyya=metrics['bhattacharyya'],
            max_weight=metrics['max_weight'],
            diversity=metrics['diversity'],
            validity_rate=metrics['validity_rate'],
            novelty_score=metrics['novelty_score'],
            converged=converged,
            failed_metrics=failed_metrics,
            recommendations=recommendations,
        )


def print_convergence_report(metrics: ConvergenceMetrics):
    """
    Print a formatted convergence report
    
    Args:
        metrics: ConvergenceMetrics object
    """
    print("\n" + "="*60)
    print("CONVERGENCE MONITORING REPORT")
    print("="*60)
    
    print("\nMetrics:")
    print(f"  Hysteresis:        {metrics.hysteresis:.4f} (threshold: ≤2.0)")
    print(f"  Bhattacharyya:     {metrics.bhattacharyya:.4f} (threshold: ≤0.03)")
    print(f"  Max Weight:        {metrics.max_weight:.4f} (threshold: ≤0.05)")
    print(f"  Diversity:         {metrics.diversity:.4f} (threshold: ≥0.7)")
    print(f"  Validity Rate:     {metrics.validity_rate:.4f} (threshold: ≥0.95)")
    print(f"  Novelty Score:     {metrics.novelty_score:.4f} (threshold: ≥0.5)")
    
    print(f"\nConvergence Status: {'✓ CONVERGED' if metrics.converged else '✗ NOT CONVERGED'}")
    
    if metrics.failed_metrics:
        print(f"\nFailed Metrics: {', '.join(metrics.failed_metrics)}")
    
    if metrics.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(metrics.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("="*60 + "\n")
