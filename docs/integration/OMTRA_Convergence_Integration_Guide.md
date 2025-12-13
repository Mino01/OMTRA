# OMTRA Convergence Monitoring Integration Guide

**Author:** Manus AI  
**Date:** December 11, 2025  
**Version:** 1.0

---

## Overview

This guide explains how to integrate the new convergence monitoring module into OMTRA workflows. The convergence monitoring system is based on Free Energy Perturbation (FEP) best practices from Martin Olsson's research, adapted for molecular generation tasks.

---

## What is Convergence Monitoring?

Convergence monitoring assesses the quality and reliability of molecular generation by measuring:

- **Hysteresis**: Consistency of the generation process
- **Bhattacharyya Coefficient**: Distribution overlap and sampling quality
- **Maximum Weight**: Detection of poor sampling (single sample dominance)
- **Diversity**: Molecular diversity in generated set
- **Validity Rate**: Chemical validity of generated molecules
- **Novelty Score**: Novelty compared to reference molecules

These metrics provide confidence in generation results and actionable recommendations for improvement.

---

## Installation

The convergence monitoring module is already included in OMTRA at `/home/ubuntu/OMTRA/omtra/convergence/`.

**Requirements:**
- NumPy (already installed)
- RDKit (optional, for diversity/validity calculations)

To install RDKit:
```bash
conda install -c conda-forge rdkit
# or
pip install rdkit
```

---

## Quick Start

### Basic Usage

```python
from omtra.convergence.monitor import ConvergenceMonitor, print_convergence_report

# Initialize monitor
monitor = ConvergenceMonitor()

# Your generated molecules
molecules = ["CCO", "CC(C)O", "CCCO", ...]  # SMILES strings

# Run monitoring
metrics = monitor.monitor(
    molecules=molecules,
    task_type='denovo_ligand_condensed'
)

# Print report
print_convergence_report(metrics)

# Save metrics
metrics.save('convergence_metrics.json')
```

### With Energy Trajectories

```python
import numpy as np

# Forward and backward trajectories
forward_energies = np.array([...])
backward_energies = np.array([...])

metrics = monitor.monitor(
    molecules=molecules,
    forward_energies=forward_energies,
    backward_energies=backward_energies,
    task_type='protein_conditioned_ligand'
)
```

### With Novelty Assessment

```python
# Reference molecules (e.g., training set)
reference_molecules = ["c1ccccc1", "CCO", ...]

metrics = monitor.monitor(
    molecules=molecules,
    reference_molecules=reference_molecules,
    task_type='denovo_ligand_condensed'
)
```

---

## Integration with OMTRA Tasks

### Modifying Task Classes

To integrate convergence monitoring into OMTRA task classes, add monitoring after generation:

```python
# In omtra/tasks/denovo_ligand_condensed.py

from omtra.convergence.monitor import ConvergenceMonitor, print_convergence_report

class DeNovoLigandCondensed:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = ConvergenceMonitor()
    
    def run(self):
        # ... existing generation code ...
        
        # Extract SMILES from results
        molecules = [mol['smiles'] for mol in results['molecules']]
        
        # Run convergence monitoring
        metrics = self.monitor.monitor(
            molecules=molecules,
            task_type='denovo_ligand_condensed'
        )
        
        # Print report
        print_convergence_report(metrics)
        
        # Save metrics
        metrics.save(self.output_dir / 'convergence_metrics.json')
        
        # Add metrics to results
        results['convergence'] = metrics.to_dict()
        
        return results
```

### CLI Integration

Add convergence monitoring flag to CLI:

```python
# In cli.py

parser.add_argument(
    "--monitor_convergence",
    action="store_true",
    help="Enable convergence monitoring"
)

# In main execution
if args.monitor_convergence:
    from omtra.convergence.monitor import ConvergenceMonitor, print_convergence_report
    
    monitor = ConvergenceMonitor()
    metrics = monitor.monitor(
        molecules=generated_molecules,
        task_type=args.task
    )
    print_convergence_report(metrics)
```

---

## Metrics Explained

### Hysteresis

**What it measures:** Consistency of the generation process by comparing forward and backward trajectories.

**Interpretation:**
- **Low values (< 2.0):** Good consistency, reliable generation
- **High values (> 2.0):** Inconsistent generation, may need more timesteps

**How to improve:**
- Increase `n_timesteps` (e.g., from 250 to 500)
- Use deterministic sampling (`stochastic_sampling=False`)
- Increase model checkpoint quality

### Bhattacharyya Coefficient

**What it measures:** Overlap between two distributions (e.g., first half vs second half of generation).

**Interpretation:**
- **Low values (< 0.03):** Good overlap, consistent sampling
- **High values (> 0.03):** Poor overlap, inconsistent sampling

**How to improve:**
- Run multiple independent generations
- Increase `n_replicates`
- Use ensemble averaging

### Maximum Weight

**What it measures:** Whether a single sample dominates the ensemble.

**Interpretation:**
- **Low values (< 0.05):** Good sampling diversity
- **High values (> 0.05):** Single sample dominates, poor sampling

**How to improve:**
- Enable `stochastic_sampling`
- Increase `noise_scaler`
- Increase temperature parameter

### Diversity

**What it measures:** Molecular diversity using Tanimoto similarity.

**Interpretation:**
- **High values (> 0.7):** Good diversity
- **Low values (< 0.7):** Low diversity, similar molecules

**How to improve:**
- Increase `n_samples`
- Use different random seeds
- Adjust sampling parameters

### Validity Rate

**What it measures:** Fraction of chemically valid molecules.

**Interpretation:**
- **High values (> 0.95):** Most molecules are valid
- **Low values (< 0.95):** Many invalid molecules

**How to improve:**
- Check model checkpoint quality
- Use post-processing filters
- Add validity constraints during generation

### Novelty Score

**What it measures:** Novelty compared to reference set (e.g., training data).

**Interpretation:**
- **High values (> 0.5):** Novel molecules
- **Low values (< 0.5):** Similar to training set

**How to improve:**
- Adjust sampling parameters
- Use conditional generation with novel constraints
- Increase exploration (higher temperature)

---

## Custom Thresholds

You can customize convergence thresholds based on your requirements:

```python
custom_thresholds = {
    'hysteresis': 1.0,  # More strict (default: 2.0)
    'bhattacharyya': 0.02,  # More strict (default: 0.03)
    'max_weight': 0.03,  # More strict (default: 0.05)
    'diversity': 0.8,  # More strict (default: 0.7)
    'validity_rate': 0.98,  # More strict (default: 0.95)
    'novelty_score': 0.6,  # More strict (default: 0.5)
}

monitor = ConvergenceMonitor(thresholds=custom_thresholds)
```

---

## Example Workflows

### Workflow 1: Basic Generation with Monitoring

```bash
# Run OMTRA with convergence monitoring
python cli.py \
  --task denovo_ligand_condensed \
  --n_samples 100 \
  --n_timesteps 250 \
  --monitor_convergence \
  --output_dir ./output
```

### Workflow 2: Protein-Conditioned with Trajectories

```python
from omtra.tasks.protein_conditioned_ligand import ProteinConditionedLigand
from omtra.convergence.monitor import ConvergenceMonitor, print_convergence_report

# Run generation
task = ProteinConditionedLigand(
    checkpoint='path/to/checkpoint',
    protein_path='protein.pdb',
    n_samples=50,
    n_timesteps=250,
    output_dir='./output'
)

results = task.run()

# Monitor convergence
monitor = ConvergenceMonitor()
metrics = monitor.monitor(
    molecules=[mol['smiles'] for mol in results['molecules']],
    forward_energies=results.get('forward_energies'),
    backward_energies=results.get('backward_energies'),
    task_type='protein_conditioned_ligand'
)

print_convergence_report(metrics)
```

### Workflow 3: Iterative Improvement

```python
monitor = ConvergenceMonitor()

# Initial generation
molecules = generate_molecules(n_samples=50, n_timesteps=250)
metrics = monitor.monitor(molecules=molecules)

# Check convergence
if not metrics.converged:
    print("Not converged. Applying recommendations...")
    
    # Apply recommendations
    if 'hysteresis' in metrics.failed_metrics:
        # Increase timesteps
        molecules = generate_molecules(n_samples=50, n_timesteps=500)
    
    if 'diversity' in metrics.failed_metrics:
        # Increase samples
        molecules = generate_molecules(n_samples=100, n_timesteps=250)
    
    # Re-check convergence
    metrics = monitor.monitor(molecules=molecules)

print_convergence_report(metrics)
```

---

## Output Format

### JSON Output

```json
{
  "hysteresis": 0.45,
  "bhattacharyya": 0.02,
  "max_weight": 0.03,
  "diversity": 0.75,
  "validity_rate": 0.97,
  "novelty_score": 0.68,
  "converged": true,
  "failed_metrics": [],
  "recommendations": []
}
```

### Console Output

```
============================================================
CONVERGENCE MONITORING REPORT
============================================================

Metrics:
  Hysteresis:        0.4500 (threshold: ≤2.0)
  Bhattacharyya:     0.0200 (threshold: ≤0.03)
  Max Weight:        0.0300 (threshold: ≤0.05)
  Diversity:         0.7500 (threshold: ≥0.7)
  Validity Rate:     0.9700 (threshold: ≥0.95)
  Novelty Score:     0.6800 (threshold: ≥0.5)

Convergence Status: ✓ CONVERGED

============================================================
```

---

## Best Practices

### 1. Always Monitor Important Generations

For production use, always enable convergence monitoring to ensure reliability:

```python
# Production generation
metrics = monitor.monitor(molecules=molecules, task_type=task_type)
if not metrics.converged:
    # Log warning or retry with improved parameters
    logger.warning(f"Generation not converged: {metrics.failed_metrics}")
```

### 2. Use Appropriate Thresholds

Adjust thresholds based on your use case:

- **Drug discovery:** Strict thresholds (high validity, novelty)
- **Exploration:** Relaxed thresholds (prioritize diversity)
- **Optimization:** Strict convergence (low hysteresis, good sampling)

### 3. Combine Multiple Metrics

Don't rely on a single metric. Use the full suite for comprehensive assessment:

```python
if metrics.converged:
    print("All metrics passed!")
else:
    print(f"Failed: {metrics.failed_metrics}")
    print(f"Recommendations: {metrics.recommendations}")
```

### 4. Save Metrics for Analysis

Always save metrics for later analysis and comparison:

```python
metrics.save(output_dir / 'convergence_metrics.json')
```

### 5. Iterate Based on Recommendations

Use the recommendations to improve generation:

```python
for rec in metrics.recommendations:
    print(f"Recommendation: {rec}")
    # Apply recommendation and re-run
```

---

## Troubleshooting

### RDKit Not Available

If RDKit is not installed, diversity and validity calculations will be skipped:

```
Warning: RDKit not available, skipping diversity calculation
```

**Solution:** Install RDKit:
```bash
conda install -c conda-forge rdkit
```

### All Metrics Fail

If all metrics fail, check:

1. **Molecule quality:** Are SMILES valid?
2. **Sample size:** Is `n_samples` sufficient (≥10)?
3. **Generation parameters:** Are `n_timesteps`, `noise_scaler` appropriate?

### High Hysteresis

If hysteresis is consistently high:

1. Increase `n_timesteps` (e.g., 250 → 500)
2. Check model checkpoint quality
3. Use deterministic sampling

### Low Diversity

If diversity is consistently low:

1. Increase `n_samples`
2. Enable `stochastic_sampling`
3. Use different random seeds
4. Increase temperature/noise

---

## API Reference

### ConvergenceMonitor

```python
class ConvergenceMonitor:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None)
    
    def monitor(
        self,
        molecules: List[str],
        energies: Optional[np.ndarray] = None,
        forward_energies: Optional[np.ndarray] = None,
        backward_energies: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        reference_molecules: Optional[List[str]] = None,
        task_type: str = 'denovo'
    ) -> ConvergenceMetrics
```

### ConvergenceMetrics

```python
@dataclass
class ConvergenceMetrics:
    hysteresis: float
    bhattacharyya: float
    max_weight: float
    diversity: float
    validity_rate: float
    novelty_score: float
    converged: bool
    failed_metrics: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict
    def save(self, path: Path)
```

### Utility Functions

```python
def print_convergence_report(metrics: ConvergenceMetrics)
```

---

## Examples

See `/home/ubuntu/OMTRA/examples/convergence_monitoring_example.py` for complete examples:

1. Basic convergence monitoring
2. Monitoring with forward/backward trajectories
3. Novelty assessment with reference set
4. Poor convergence detection
5. Custom convergence thresholds

Run examples:
```bash
cd /home/ubuntu/OMTRA
python examples/convergence_monitoring_example.py
```

---

## Conclusion

The convergence monitoring module provides robust quality assessment for OMTRA molecular generation. By integrating these metrics into your workflows, you can:

- **Ensure reliability** of generated molecules
- **Detect issues** early in the generation process
- **Improve results** with actionable recommendations
- **Build confidence** in production deployments

For questions or issues, refer to the OMTRA documentation or contact the development team.

---

## References

[1] Martin A. Olsson, "QM/MM free-energy perturbation and other methods to estimate ligand-binding affinities," PhD Thesis, Lund University, 2016.

[2] Martin A. Olsson and Ulf Ryde, "Comparison of QM/MM Methods To Obtain Ligand-Binding Free Energies," Journal of Chemical Theory and Computation, 2017.

[3] ForcelabElixir FEP Module Documentation, 2025.
