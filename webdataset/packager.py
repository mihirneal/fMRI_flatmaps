"""WebDataset packaging for flat map data."""

import tarfile
import json
import io
from pathlib import Path
from typing import Dict, List, Iterator, Optional
import numpy as np


class WebDatasetPackager:
    """Packages flat map data into WebDataset tar files."""
    
    def __init__(self, output_dir: str | Path, max_samples_per_shard: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_samples_per_shard = max_samples_per_shard
    
    def package_samples(self, samples: List[Dict], dataset_name: str) -> List[Path]:
        """Package list of samples into WebDataset shards.
        
        Args:
            samples: List of sample dictionaries from FlatMapGenerator
            dataset_name: Base name for dataset shards
            
        Returns:
            List of created shard file paths
        """
        shard_files = []
        
        # Split samples into shards
        n_shards = (len(samples) + self.max_samples_per_shard - 1) // self.max_samples_per_shard
        
        for shard_idx in range(n_shards):
            start_idx = shard_idx * self.max_samples_per_shard
            end_idx = min((shard_idx + 1) * self.max_samples_per_shard, len(samples))
            shard_samples = samples[start_idx:end_idx]
            
            shard_file = self.output_dir / f"{dataset_name}_{shard_idx:04d}.tar"
            self._create_shard(shard_samples, shard_file)
            shard_files.append(shard_file)
            
            print(f"Created shard {shard_file} with {len(shard_samples)} samples")
        
        return shard_files
    
    def _create_shard(self, samples: List[Dict], shard_file: Path):
        """Create a single WebDataset shard tar file."""
        with tarfile.open(shard_file, 'w') as tar:
            for sample in samples:
                key = sample['__key__']
                
                # Add each component of the sample
                for ext, data in sample.items():
                    if ext == '__key__':
                        continue
                    
                    filename = f"{key}.{ext}"
                    
                    if ext == 'bold.npy':
                        # Save numpy array
                        buffer = io.BytesIO()
                        np.save(buffer, data)
                        buffer.seek(0)
                        self._add_buffer_to_tar(tar, filename, buffer)
                    
                    elif ext == 'mask.npz':
                        # Save sparse mask in npz format
                        buffer = io.BytesIO()
                        np.savez_compressed(buffer, **data)
                        buffer.seek(0)
                        self._add_buffer_to_tar(tar, filename, buffer)
                    
                    elif ext.endswith('.json'):
                        # Save JSON data
                        json_str = json.dumps(data, indent=2)
                        buffer = io.BytesIO(json_str.encode('utf-8'))
                        self._add_buffer_to_tar(tar, filename, buffer)
                    
                    else:
                        print(f"Warning: Unknown extension {ext}, skipping")
    
    def _add_buffer_to_tar(self, tar: tarfile.TarFile, filename: str, buffer: io.BytesIO):
        """Add a buffer to tar file."""
        tarinfo = tarfile.TarInfo(name=filename)
        tarinfo.size = len(buffer.getvalue())
        tar.addfile(tarinfo, buffer)
    
    def create_dataset_from_generator(self, 
                                    sample_generator: Iterator[Dict],
                                    dataset_name: str,
                                    buffer_size: Optional[int] = None) -> List[Path]:
        """Create WebDataset from a generator of samples.
        
        Args:
            sample_generator: Generator yielding sample dictionaries
            dataset_name: Base name for dataset
            buffer_size: Number of samples to buffer before writing (default: max_samples_per_shard)
            
        Returns:
            List of created shard file paths
        """
        if buffer_size is None:
            buffer_size = self.max_samples_per_shard
        
        shard_files = []
        buffer = []
        shard_idx = 0
        
        for sample in sample_generator:
            buffer.append(sample)
            
            if len(buffer) >= buffer_size:
                shard_file = self.output_dir / f"{dataset_name}_{shard_idx:04d}.tar"
                self._create_shard(buffer, shard_file)
                shard_files.append(shard_file)
                
                print(f"Created shard {shard_file} with {len(buffer)} samples")
                
                buffer = []
                shard_idx += 1
        
        # Write remaining samples
        if buffer:
            shard_file = self.output_dir / f"{dataset_name}_{shard_idx:04d}.tar"
            self._create_shard(buffer, shard_file)
            shard_files.append(shard_file)
            
            print(f"Created final shard {shard_file} with {len(buffer)} samples")
        
        return shard_files
    
    def validate_shard(self, shard_file: Path) -> bool:
        """Validate that a shard file is properly formatted."""
        try:
            with tarfile.open(shard_file, 'r') as tar:
                members = tar.getmembers()
                
                # Group by key
                keys = {}
                for member in members:
                    if '.' not in member.name:
                        continue
                    
                    key, ext = member.name.rsplit('.', 1)
                    if key not in keys:
                        keys[key] = set()
                    keys[key].add(ext)
                
                # Check that each sample has required components
                required_extensions = {'bold.npy', 'mask.npz', 'events.json', 'meta.json'}
                
                for key, extensions in keys.items():
                    missing = required_extensions - extensions
                    if missing:
                        print(f"Sample {key} missing extensions: {missing}")
                        return False
                
                print(f"Shard {shard_file} is valid with {len(keys)} samples")
                return True
                
        except Exception as e:
            print(f"Error validating shard {shard_file}: {e}")
            return False
    
    def get_dataset_stats(self, shard_files: List[Path]) -> Dict:
        """Get statistics about the created dataset."""
        total_samples = 0
        total_size_mb = 0
        
        for shard_file in shard_files:
            if shard_file.exists():
                size_mb = shard_file.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb
                
                # Count samples in shard
                try:
                    with tarfile.open(shard_file, 'r') as tar:
                        members = tar.getmembers()
                        keys = set()
                        for member in members:
                            if '.' in member.name:
                                key = member.name.rsplit('.', 1)[0]
                                keys.add(key)
                        total_samples += len(keys)
                except:
                    pass
        
        return {
            'n_shards': len(shard_files),
            'total_samples': total_samples,
            'total_size_mb': round(total_size_mb, 2),
            'avg_samples_per_shard': total_samples // len(shard_files) if shard_files else 0,
            'avg_mb_per_shard': round(total_size_mb / len(shard_files), 2) if shard_files else 0
        }