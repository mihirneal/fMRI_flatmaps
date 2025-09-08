#!/usr/bin/env python3
"""
Visualization script for comparing flat map datasets.

Usage:
    python visualize_flat_maps.py --hcp_data datasets/fmri-fm.datasets/hcp-flat/hcp-flat_0001.tar --ampa_data datasets/AMPA_flat/AMPA_flat_0000.tar
"""

import argparse
import tarfile
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def load_webdataset_sample(tar_path: Path, sample_key: str = None) -> Dict:
    """Load a sample from WebDataset tar file."""
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        
        # Get available sample keys
        keys = set()
        for member in members:
            if '.' in member.name:
                key = member.name.rsplit('.', 1)[0]
                keys.add(key)
        
        # Use first key if none specified
        if sample_key is None:
            if not keys:
                raise ValueError(f"No samples found in {tar_path}")
            sample_key = sorted(keys)[0]
        
        print(f"Loading sample: {sample_key}")
        
        # Load all components for this sample
        sample = {}
        for member in members:
            if not member.name.startswith(sample_key + '.'):
                continue
                
            ext = member.name.split('.')[-1]
            data = tar.extractfile(member).read()
            
            if ext == 'npy':
                sample['bold'] = np.load(io.BytesIO(data))
            elif ext == 'npz':
                npz_data = np.load(io.BytesIO(data))
                sample['mask'] = {
                    'data': npz_data['data'],
                    'row': npz_data['row'], 
                    'col': npz_data['col'],
                    'shape': npz_data['shape']
                }
            elif ext == 'json':
                if 'meta' in member.name:
                    sample['meta'] = json.loads(data.decode('utf-8'))
                elif 'events' in member.name:
                    sample['events'] = json.loads(data.decode('utf-8'))
        
        return sample


def reconstruct_flat_image(bold_data: np.ndarray, mask_data: Dict) -> np.ndarray:
    """Reconstruct 2D flat image from flattened data and sparse mask."""
    # Reconstruct sparse mask
    sparse_mask = scipy.sparse.coo_matrix(
        (mask_data['data'], (mask_data['row'], mask_data['col'])),
        shape=tuple(mask_data['shape'])
    ).toarray().astype(bool)
    
    T, D = bold_data.shape
    H, W = sparse_mask.shape
    
    # Reconstruct flat images
    flat_images = np.zeros((T, H, W))
    for t in range(T):
        flat_images[t][sparse_mask] = bold_data[t]
    
    return flat_images, sparse_mask


def compute_flat_map_stats(flat_images: np.ndarray, mask: np.ndarray) -> Dict:
    """Compute statistics for flat map data."""
    valid_data = flat_images[:, mask]
    
    return {
        'shape': flat_images.shape,
        'n_timepoints': flat_images.shape[0],
        'n_valid_pixels': mask.sum(),
        'coverage': mask.sum() / mask.size,
        'mean_signal': valid_data.mean(),
        'std_signal': valid_data.std(),
        'min_signal': valid_data.min(),
        'max_signal': valid_data.max(),
        'temporal_mean': valid_data.mean(axis=0).mean(),
        'temporal_std': valid_data.mean(axis=0).std()
    }


def plot_flat_map_comparison(hcp_sample: Dict, ampa_sample: Dict, output_path: Path = None):
    """Create comprehensive comparison plots."""
    
    # Reconstruct flat images
    hcp_images, hcp_mask = reconstruct_flat_image(hcp_sample['bold'], hcp_sample['mask'])
    ampa_images, ampa_mask = reconstruct_flat_image(ampa_sample['bold'], ampa_sample['mask'])
    
    # Compute stats
    hcp_stats = compute_flat_map_stats(hcp_images, hcp_mask)
    ampa_stats = compute_flat_map_stats(ampa_images, ampa_mask)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Masks comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(hcp_mask, cmap='gray')
    ax1.set_title(f'HCP Mask\\n{hcp_mask.sum()} pixels')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ampa_mask, cmap='gray')
    ax2.set_title(f'AMPA Mask\\n{ampa_mask.sum()} pixels')
    ax2.axis('off')
    
    # 2. Mean activation maps
    hcp_mean = hcp_images.mean(axis=0)
    ampa_mean = ampa_images.mean(axis=0)
    
    # Use same color scale
    vmin = min(hcp_mean[hcp_mask].min(), ampa_mean[ampa_mask].min())
    vmax = max(hcp_mean[hcp_mask].max(), ampa_mean[ampa_mask].max())
    
    ax3 = fig.add_subplot(gs[0, 2])
    im1 = ax3.imshow(hcp_mean, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax3.set_title('HCP Mean Activation')
    ax3.axis('off')
    plt.colorbar(im1, ax=ax3, shrink=0.6)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im2 = ax4.imshow(ampa_mean, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax4.set_title('AMPA Mean Activation')
    ax4.axis('off')
    plt.colorbar(im2, ax=ax4, shrink=0.6)
    
    # 3. Standard deviation maps
    hcp_std = hcp_images.std(axis=0)
    ampa_std = ampa_images.std(axis=0)
    
    std_vmax = max(hcp_std[hcp_mask].max(), ampa_std[ampa_mask].max())
    
    ax5 = fig.add_subplot(gs[1, 0])
    im3 = ax5.imshow(hcp_std, cmap='viridis', vmin=0, vmax=std_vmax)
    ax5.set_title('HCP Std Deviation')
    ax5.axis('off')
    plt.colorbar(im3, ax=ax5, shrink=0.6)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im4 = ax6.imshow(ampa_std, cmap='viridis', vmin=0, vmax=std_vmax)
    ax6.set_title('AMPA Std Deviation')
    ax6.axis('off')
    plt.colorbar(im4, ax=ax6, shrink=0.6)
    
    # 4. Sample timepoints
    n_samples = min(4, hcp_images.shape[0], ampa_images.shape[0])
    time_indices = np.linspace(0, min(hcp_images.shape[0], ampa_images.shape[0])-1, n_samples).astype(int)
    
    for i, t in enumerate(time_indices):
        # HCP timepoints
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(hcp_images[t], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'HCP t={t}')
        ax.axis('off')
        
        # AMPA timepoints  
        ax = fig.add_subplot(gs[3, i])
        im = ax.imshow(ampa_images[t], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'AMPA t={t}')
        ax.axis('off')
    
    # Add statistics text
    stats_text = f"""
    HCP Stats:
    Shape: {hcp_stats['shape']}
    Valid pixels: {hcp_stats['n_valid_pixels']} ({hcp_stats['coverage']:.1%})
    Signal range: [{hcp_stats['min_signal']:.3f}, {hcp_stats['max_signal']:.3f}]
    Mean ± Std: {hcp_stats['mean_signal']:.3f} ± {hcp_stats['std_signal']:.3f}
    
    AMPA Stats:
    Shape: {ampa_stats['shape']}
    Valid pixels: {ampa_stats['n_valid_pixels']} ({ampa_stats['coverage']:.1%})
    Signal range: [{ampa_stats['min_signal']:.3f}, {ampa_stats['max_signal']:.3f}]
    Mean ± Std: {ampa_stats['mean_signal']:.3f} ± {ampa_stats['std_signal']:.3f}
    """
    
    fig.text(0.02, 0.35, stats_text, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.suptitle('Flat Map Comparison: HCP vs AMPA', fontsize=16, y=0.95)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    plt.show()


def plot_temporal_comparison(hcp_sample: Dict, ampa_sample: Dict):
    """Plot temporal signal characteristics."""
    
    # Reconstruct flat images
    hcp_images, hcp_mask = reconstruct_flat_image(hcp_sample['bold'], hcp_sample['mask'])
    ampa_images, ampa_mask = reconstruct_flat_image(ampa_sample['bold'], ampa_sample['mask'])
    
    # Extract mean timeseries over valid pixels
    hcp_ts = hcp_images[:, hcp_mask].mean(axis=1)
    ampa_ts = ampa_images[:, ampa_mask].mean(axis=1)
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Timeseries
    ax1.plot(hcp_ts, label='HCP', alpha=0.8)
    ax1.plot(ampa_ts, label='AMPA', alpha=0.8)
    ax1.set_xlabel('Time (TRs)')
    ax1.set_ylabel('Mean Signal')
    ax1.set_title('Average Timeseries Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Signal distributions
    hcp_valid = hcp_images[:, hcp_mask].flatten()
    ampa_valid = ampa_images[:, ampa_mask].flatten()
    
    ax2.hist(hcp_valid, bins=50, alpha=0.6, label='HCP', density=True)
    ax2.hist(ampa_valid, bins=50, alpha=0.6, label='AMPA', density=True)
    ax2.set_xlabel('Signal Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Signal Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Correlation plot
    min_len = min(len(hcp_ts), len(ampa_ts))
    ax3.scatter(hcp_ts[:min_len], ampa_ts[:min_len], alpha=0.6)
    ax3.set_xlabel('HCP Mean Signal')
    ax3.set_ylabel('AMPA Mean Signal') 
    ax3.set_title('Temporal Correlation')
    
    # Add correlation coefficient
    corr = np.corrcoef(hcp_ts[:min_len], ampa_ts[:min_len])[0, 1]
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle="round", facecolor="white"))
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare flat map datasets")
    
    parser.add_argument("--hcp_data", required=True, help="Path to HCP WebDataset tar file")
    parser.add_argument("--ampa_data", required=True, help="Path to AMPA WebDataset tar file")
    parser.add_argument("--hcp_sample", help="Specific HCP sample key to load")
    parser.add_argument("--ampa_sample", help="Specific AMPA sample key to load")
    parser.add_argument("--output", help="Output path for comparison plot")
    
    args = parser.parse_args()
    
    # Load samples
    print("Loading HCP sample...")
    hcp_sample = load_webdataset_sample(Path(args.hcp_data), args.hcp_sample)
    
    print("Loading AMPA sample...")  
    ampa_sample = load_webdataset_sample(Path(args.ampa_data), args.ampa_sample)
    
    # Create visualizations
    print("Creating comparison plots...")
    plot_flat_map_comparison(hcp_sample, ampa_sample, 
                            Path(args.output) if args.output else None)
    
    print("Creating temporal comparison...")
    plot_temporal_comparison(hcp_sample, ampa_sample)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()