#!/usr/bin/env python3
"""
Quick flat map test using template-based approach (no FreeSurfer needed).
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import json
from scipy import ndimage
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from bids.loader import BIDSLoader
from webdataset.packager import WebDatasetPackager


def create_template_flat_map(func_data: np.ndarray, output_size=(256, 256)) -> tuple:
    """Create a simple flat map using template-based approach."""
    
    # Simple brain mask (exclude background)
    brain_mask = func_data.mean(axis=-1) > func_data.mean() * 0.1
    
    # Find brain bounding box
    coords = np.where(brain_mask)
    if len(coords[0]) == 0:
        return np.zeros((func_data.shape[-1], *output_size)), np.zeros(output_size, dtype=bool)
        
    min_coords = [c.min() for c in coords]
    max_coords = [c.max() for c in coords]
    
    # Extract brain region
    brain_region = func_data[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1, 
        min_coords[2]:max_coords[2]+1
    ]
    
    # Create "flat" projection by averaging over depth (Z axis)
    flat_projection = brain_region.mean(axis=2)  # (X, Y, T)
    flat_projection = np.transpose(flat_projection, (2, 0, 1))  # (T, X, Y)
    
    # Resize to target size
    T, H, W = flat_projection.shape
    target_H, target_W = output_size
    
    # Simple resize using scipy
    flat_resized = np.zeros((T, target_H, target_W))
    mask_resized = np.zeros((target_H, target_W), dtype=bool)
    
    # Create a simple oval mask for brain-like shape
    y_center, x_center = target_H // 2, target_W // 2
    y_radius, x_radius = target_H // 3, target_W // 2.2
    
    y_grid, x_grid = np.mgrid[0:target_H, 0:target_W]
    mask_resized = ((y_grid - y_center) / y_radius) ** 2 + ((x_grid - x_center) / x_radius) ** 2 <= 1
    
    # Interpolate data to fit mask
    for t in range(T):
        # Simple interpolation
        resized = ndimage.zoom(flat_projection[t], 
                              (target_H / H, target_W / W), 
                              order=1, mode='constant', cval=0)
        flat_resized[t] = resized * mask_resized
    
    return flat_resized, mask_resized


def process_single_subject_quick(bids_dir: Path, subject_id: str, output_dir: Path):
    """Quick processing of single subject without FreeSurfer."""
    
    loader = BIDSLoader(bids_dir)
    subject = loader.get_subject(subject_id)
    
    # Get functional files
    func_files = subject.get_functional_files()
    if not func_files:
        raise ValueError(f"No functional files found for {subject_id}")
    
    func_file = func_files[0]  # Use first run
    print(f"Processing {func_file.name}...")
    
    # Load functional data
    func_data, func_header, func_metadata = subject.load_functional_data(func_file)
    print(f"Functional data shape: {func_data.shape}")
    
    # Create template flat map
    flat_images, flat_mask = create_template_flat_map(func_data)
    print(f"Flat map shape: {flat_images.shape}, mask pixels: {flat_mask.sum()}")
    
    # Flatten data for storage (like the real pipeline)
    T, H, W = flat_images.shape
    valid_pixels = flat_mask.sum()
    flattened_data = np.zeros((T, valid_pixels), dtype=np.float32)
    
    for t in range(T):
        flattened_data[t] = flat_images[t][flat_mask]
    
    # Create sparse mask
    y_coords, x_coords = np.where(flat_mask)
    sparse_mask_data = np.ones(len(y_coords), dtype=np.uint8)
    sparse_mask = sp.coo_matrix((sparse_mask_data, (y_coords, x_coords)), shape=(H, W))
    
    # Load events if available
    events_file = subject.get_events_file(func_file)
    events = []
    if events_file and events_file.exists():
        events_df = subject.load_events(events_file)
        events = events_df.to_dict('records')
    
    # Create metadata
    metadata = {
        'subject_id': subject_id,
        'func_file': str(func_file.name),
        'shape': (H, W),
        'n_timepoints': T,
        'processing': 'template_based_quick',
        **func_metadata
    }
    
    # Package sample
    sample = {
        'bold.npy': flattened_data,
        'mask.npz': {
            'data': sparse_mask.data,
            'row': sparse_mask.row,
            'col': sparse_mask.col,
            'shape': sparse_mask.shape
        },
        'events.json': events,
        'meta.json': metadata,
        '__key__': f"{subject_id}_{func_file.stem}"
    }
    
    # Create output directory and package
    output_dir.mkdir(exist_ok=True, parents=True)
    packager = WebDatasetPackager(output_dir)
    shard_files = packager.package_samples([sample], "AMPA_quick")
    
    print(f"Created WebDataset: {shard_files[0]}")
    return shard_files[0]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick flat map test")
    parser.add_argument("--bids_dir", required=True, help="BIDS directory")
    parser.add_argument("--output_dir", required=True, help="Output directory") 
    parser.add_argument("--subject", default="sub-agematch01", help="Subject ID")
    
    args = parser.parse_args()
    
    shard_file = process_single_subject_quick(
        Path(args.bids_dir), 
        args.subject,
        Path(args.output_dir)
    )
    
    print(f"\\nQuick processing complete!")
    print(f"Output: {shard_file}")
    print(f"\\nTo visualize, you can inspect the data with:")
    print(f"python -c \"")
    print(f"import tarfile, numpy as np, json, io")
    print(f"with tarfile.open('{shard_file}') as tar:")
    print(f"    bold = np.load(io.BytesIO(tar.extractfile('{args.subject}_task-rest_run-01_bold.bold.npy').read()))")
    print(f"    print('Shape:', bold.shape)")
    print(f"\"")