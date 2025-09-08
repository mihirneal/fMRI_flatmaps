# fMRI Flat Map Preprocessing Pipeline

Preprocessing pipeline for converting BIDS-formatted fMRI data into flat map representations. Worked on by Mihir Tripathy (Baylor College of Medicine) and Claude 4.0 Sonnet.

## Overview

The pipeline performs these key steps:
1. **Surface Reconstruction**: Uses FreeSurfer to create cortical surface meshes
2. **Surface Flattening**: Converts 3D surfaces to 2D flat coordinates 
3. **Volume-to-Surface Projection**: Maps fMRI data from 3D volumes to surface vertices
4. **Flat Map Generation**: Creates 2D flat map images from surface data
5. **WebDataset Packaging**: Packages data into efficient tar-based format

## Requirements

### Software Dependencies
- **FreeSurfer** (version 6.0+): Required for cortical surface reconstruction
  - Download from: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
  - Ensure `FREESURFER_HOME` environment variable is set

### Python Dependencies
Install with:
```bash
uv sync
```

## Installation

1. Ensure FreeSurfer is installed and configured
2. Install Python dependencies:
   ```bash
   cd fMRI_flatmaps
   uv sync
   ```

## Usage

### Basic Usage

```bash
python process_bids_to_flatmaps.py \\
    --bids_dir /path/to/your/AMPA_DS_bids \\
    --output_dir /path/to/output/flatmaps \\
    --subjects_dir /path/to/freesurfer/subjects
```

### Complete Example

```bash
python process_bids_to_flatmaps.py \\
    --bids_dir ../datasets/AMPA_DS_bids \\
    --output_dir ../datasets/AMPA_flat \\
    --subjects_dir ./freesurfer_subjects \\
    --target_size 256 256 \\
    --sampling_method ribbon \\
    --max_samples_per_shard 50 \\
    --dataset_name AMPA_flat
```

### Options

- `--bids_dir`: Path to BIDS dataset directory
- `--output_dir`: Output directory for WebDataset shards  
- `--subjects_dir`: FreeSurfer subjects directory
- `--subjects`: Process specific subjects (default: all)
- `--task`: Process specific task (default: all tasks)
- `--target_size`: Target flat map size in pixels (default: 256 256)
- `--sampling_method`: Volume-to-surface sampling method:
  - `ribbon`: Sample along cortical ribbon (recommended)
  - `nearest`: Nearest neighbor sampling
  - `trilinear`: Trilinear interpolation
- `--max_samples_per_shard`: Samples per WebDataset tar file (default: 100)
- `--skip_freesurfer`: Skip FreeSurfer if already run
- `--dry_run`: Preview what would be processed

## Output Format

The pipeline generates WebDataset tar files containing:

- `bold.npy`: Flattened fMRI timeseries data (T, D) where D = number of cortical pixels
- `mask.npz`: Sparse mask for reconstructing 2D flat maps (H, W)  
- `events.json`: Task events from BIDS events.tsv files
- `meta.json`: Metadata including subject info, processing parameters

## Processing Time

Estimated processing times:
- **FreeSurfer reconstruction**: ~6 hours per subject
- **Flat map generation**: ~5 minutes per functional run
- **Total for 50 subjects**: ~300-350 hours

Consider using a compute cluster for large datasets.

## Integration with fMRI-FM

Once processed, use the flat maps with the main fMRI-FM model:

```python
from src.flat_data import make_flat_wds_dataset

dataset = make_flat_wds_dataset(
    url="/path/to/output/flatmaps/*.tar",
    num_frames=16,
    clipping="random",
    shuffle=True
)
```

## Troubleshooting

### FreeSurfer Issues
- Ensure `FREESURFER_HOME` is set correctly
- Check FreeSurfer license is valid
- Verify input T1w images have good quality

### Memory Issues  
- Reduce `max_samples_per_shard` for large datasets
- Process subjects individually using `--subjects` option

### Surface Reconstruction Failures
- Some subjects may fail FreeSurfer reconstruction
- Review FreeSurfer logs in `{subjects_dir}/{subject_id}/scripts/`
- Consider manual quality control of anatomical images

## Directory Structure

```
fmri_preprocessing/
├── bids/                   # BIDS data loading
│   ├── __init__.py
│   └── loader.py
├── surface/                # Surface processing  
│   ├── __init__.py
│   ├── freesurfer.py      # FreeSurfer interface
│   └── flattening.py      # Surface flattening
├── flat/                   # Flat map generation
│   ├── __init__.py
│   ├── projection.py      # Volume-to-surface projection
│   └── generator.py       # Main flat map generator
├── webdataset/            # WebDataset packaging
│   ├── __init__.py
│   └── packager.py
├── process_bids_to_flatmaps.py  # Main script
├── requirements.txt
└── README.md
```