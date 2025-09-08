"""FreeSurfer-based surface processing."""

import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import nibabel as nib
from nibabel import freesurfer


class FreeSurferProcessor:
    """Handles FreeSurfer surface reconstruction and processing."""
    
    def __init__(self, subjects_dir: str | Path, freesurfer_home: Optional[str] = None):
        self.subjects_dir = Path(subjects_dir)
        self.subjects_dir.mkdir(exist_ok=True)
        
        # Set FreeSurfer environment
        if freesurfer_home:
            os.environ['FREESURFER_HOME'] = str(freesurfer_home)
        
        if 'FREESURFER_HOME' not in os.environ:
            # Try common locations
            common_paths = ['/usr/local/freesurfer', '/opt/freesurfer']
            for path in common_paths:
                if Path(path).exists():
                    os.environ['FREESURFER_HOME'] = path
                    break
        
        os.environ['SUBJECTS_DIR'] = str(self.subjects_dir)
    
    def run_recon_all(self, t1_file: Path, subject_id: str, 
                     stages: List[str] = None, force: bool = False) -> bool:
        """Run FreeSurfer recon-all pipeline."""
        if stages is None:
            stages = ['-all']  # Full pipeline
            
        # Check if already processed
        subject_dir = self.subjects_dir / subject_id
        if not force and (subject_dir / 'surf' / 'lh.white').exists():
            print(f"Subject {subject_id} already processed, skipping...")
            return True
            
        cmd = [
            'recon-all',
            '-subject', subject_id,
            '-i', str(t1_file),
            '-sd', str(self.subjects_dir)
        ] + stages
        
        print(f"Running FreeSurfer recon-all for {subject_id}...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"FreeSurfer completed successfully for {subject_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FreeSurfer failed for {subject_id}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def load_surface(self, subject_id: str, hemi: str, surface: str = 'white') -> Tuple[np.ndarray, np.ndarray]:
        """Load surface vertices and faces."""
        surf_file = self.subjects_dir / subject_id / 'surf' / f'{hemi}.{surface}'
        if not surf_file.exists():
            raise FileNotFoundError(f"Surface file not found: {surf_file}")
            
        vertices, faces = freesurfer.read_geometry(surf_file)
        return vertices, faces
    
    def load_flat_surface(self, subject_id: str, hemi: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load flattened surface coordinates."""
        flat_file = self.subjects_dir / subject_id / 'surf' / f'{hemi}.flat'
        if not flat_file.exists():
            # Try to create flat surface
            self.create_flat_surface(subject_id, hemi)
            
        vertices, faces = freesurfer.read_geometry(flat_file)
        return vertices, faces
    
    def create_flat_surface(self, subject_id: str, hemi: str) -> bool:
        """Create flattened surface using mris_flatten."""
        cmd = [
            'mris_flatten',
            '-w', '0',  # No smoothing
            str(self.subjects_dir / subject_id / 'surf' / f'{hemi}.inflated'),
            str(self.subjects_dir / subject_id / 'surf' / f'{hemi}.flat')
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"mris_flatten failed: {e}")
            return False
    
    def surface_to_volume_transform(self, subject_id: str) -> np.ndarray:
        """Get transform from surface to volume coordinates."""
        # Load transform files
        subject_dir = self.subjects_dir / subject_id
        
        # Try to load the registration matrix
        reg_file = subject_dir / 'mri' / 'transforms' / 'talairach.xfm'
        if reg_file.exists():
            # Parse FreeSurfer transform
            with open(reg_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'Linear_Transform' in line:
                        transform_lines = lines[i+1:i+4]
                        transform = []
                        for tl in transform_lines:
                            transform.append([float(x) for x in tl.strip().split()[:3]])
                        return np.array(transform)
        
        # Default identity if no transform found
        return np.eye(3)
    
    def get_cortical_ribbon_mask(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get cortical ribbon masks for volume-to-surface mapping."""
        subject_dir = self.subjects_dir / subject_id
        
        # Load ribbon mask
        ribbon_file = subject_dir / 'mri' / 'ribbon.mgz'
        if ribbon_file.exists():
            ribbon_img = nib.load(ribbon_file)
            ribbon_data = ribbon_img.get_fdata()
            
            # FreeSurfer ribbon labels: 3=lh cortex, 42=rh cortex
            lh_mask = (ribbon_data == 3).astype(np.uint8)
            rh_mask = (ribbon_data == 42).astype(np.uint8)
            
            return lh_mask, rh_mask
        else:
            raise FileNotFoundError(f"Ribbon mask not found: {ribbon_file}")
    
    def list_subjects(self) -> List[str]:
        """List all processed subjects."""
        subjects = []
        for subject_dir in self.subjects_dir.iterdir():
            if subject_dir.is_dir() and (subject_dir / 'surf').exists():
                subjects.append(subject_dir.name)
        return subjects