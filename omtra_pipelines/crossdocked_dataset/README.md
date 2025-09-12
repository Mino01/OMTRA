# Processing the Crossdocked Dataset

## External Split Processing

To process the external splits of the Crossdocked dataset, you must use the `split_by_name.pt` file located at:

/net/galaxy/home/koes/jmgupta/omtra_2/omtra_pipelines/crossdocked_dataset/crossdocked_external_splits/split_by_name.pt

### File Structure

This file contains a PyTorch dictionary with the following structure:

- **Data Type**: Dictionary
- **Keys**: `['train', 'test']`
- **Training Set**: 100,000 samples
- **Test Set**: 100 samples

### Data Format

Each sample is a tuple containing:
1. **PDB file path** (protein structure): `*.pdb`
2. **SDF file path** (ligand structure): `*.sdf`

#### Example Data Points
```python
# Training set examples:
('DYR_STAAU_2_158_0/4xe6_X_rec_3fqc_55v_lig_tt_docked_4_pocket10.pdb', 
 'DYR_STAAU_2_158_0/4xe6_X_rec_3fqc_55v_lig_tt_docked_4.sdf')

('TRY1_BOVIN_66_246_0/1k1j_A_rec_1yp9_uiz_lig_tt_docked_1_pocket10.pdb', 
 'TRY1_BOVIN_66_246_0/1k1j_A_rec_1yp9_uiz_lig_tt_docked_1.sdf')

```
### Processing Script Overview

The main processing script is `run_crossdocked_processing_external_splits.py` which:
- Loads the external split file (`split_by_name.pt`)
- Processes protein-ligand pairs in parallel batches
- Outputs processed data to Zarr format for training and validation

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cd_directory` | `/net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types` | Crossdocked types file directory |
| `--pocket_cutoff` | `8.0` | Pocket cutoff distance (Angstroms) |
| `--zarr_output_dir` | `test_external_output.zarr` | Output directory for processed Zarr files |
| `--root_dir` | `/net/galaxy/home/koes/paf46_shared/cd2020_v1.3` | Root directory for crossdocked data |
| `--max_batches` | `None` | Maximum number of batches to process (None = all) |
| `--batch_size` | `500` | Number of ligand-receptor pairs per batch |
| `--n_cpus` | `8` | Number of CPUs for parallel processing |
| `--max_pending` | `32` | Maximum pending jobs in the processing pool |
