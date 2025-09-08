"""Flat map generation combining surface processing and projection."""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json

from bids.loader import BIDSSubject
from surface.freesurfer import FreeSurferProcessor
from surface.flattening import SurfaceFlattener
from flat.projection import VolumeToSurfaceProjector


class FlatMapGenerator:
    """Generates flat maps from BIDS fMRI data."""
    
    def __init__(self, 
                 subjects_dir: str | Path,
                 target_size: Tuple[int, int] = (256, 256),
                 sampling_method: str = 'ribbon',
                 freesurfer_home: Optional[str] = None):
        
        self.freesurfer = FreeSurferProcessor(subjects_dir, freesurfer_home)
        self.flattener = SurfaceFlattener(target_size)
        self.projector = VolumeToSurfaceProjector(sampling_method)
        self.target_size = target_size
    
    def process_subject_functional(self, 
                                 bids_subject: BIDSSubject,
                                 func_file: Path,
                                 events_file: Optional[Path] = None,
                                 run_freesurfer: bool = True) -> Dict:
        """Process a single functional run for a subject.
        
        Args:
            bids_subject: BIDSSubject instance
            func_file: Path to functional NIfTI file
            events_file: Optional events file
            run_freesurfer: Whether to run FreeSurfer if not already done
            
        Returns:
            Dictionary containing flat map data ready for WebDataset packaging
        """
        subject_id = bids_subject.subject_id
        
        # Step 1: Surface reconstruction (if needed)
        if run_freesurfer:
            anat_files = bids_subject.get_anatomical_files()
            if not anat_files:
                raise ValueError(f"No anatomical files found for {subject_id}")
            
            # Use first T1w file
            t1_file = anat_files[0]
            success = self.freesurfer.run_recon_all(t1_file, subject_id)
            if not success:
                raise RuntimeError(f"FreeSurfer failed for {subject_id}")
        
        # Step 2: Load functional data
        func_data, func_header, func_metadata = bids_subject.load_functional_data(func_file)
        func_affine = func_header.get_best_affine()
        
        # Step 3: Load surface data
        try:
            # White matter surfaces
            lh_white_verts, lh_faces = self.freesurfer.load_surface(subject_id, 'lh', 'white')
            rh_white_verts, rh_faces = self.freesurfer.load_surface(subject_id, 'rh', 'white')
            
            # Pial surfaces  
            lh_pial_verts, _ = self.freesurfer.load_surface(subject_id, 'lh', 'pial')
            rh_pial_verts, _ = self.freesurfer.load_surface(subject_id, 'rh', 'pial')
            
            # Flat surfaces
            lh_flat_verts, _ = self.freesurfer.load_flat_surface(subject_id, 'lh')
            rh_flat_verts, _ = self.freesurfer.load_flat_surface(subject_id, 'rh')
            
        except FileNotFoundError as e:
            raise RuntimeError(f"Surface files missing for {subject_id}: {e}")
        
        # Step 4: Project functional data to surfaces
        print(f"Projecting functional data to surfaces for {subject_id}...")
        
        lh_surface_data = self.projector.project_to_surface(
            func_data, func_affine, lh_white_verts, lh_pial_verts
        )
        rh_surface_data = self.projector.project_to_surface(
            func_data, func_affine, rh_white_verts, rh_pial_verts
        )
        
        # Step 5: Create flat maps
        print(f"Generating flat maps for {subject_id}...")
        
        flat_image, flat_mask = self.flattener.create_hemisphere_flat_map(
            lh_white_verts, lh_faces, lh_flat_verts, lh_surface_data,
            rh_white_verts, rh_faces, rh_flat_verts, rh_surface_data
        )
        
        # Step 6: Create sparse mask
        sparse_mask = self.flattener.create_sparse_mask(flat_mask)
        
        # Step 7: Flatten image data for storage
        # flat_image shape: (T, H, W), we want to store as (T, D) where D = number of valid pixels
        T, H, W = flat_image.shape
        valid_pixels = flat_mask.sum()
        
        # Extract data only at valid mask locations
        flattened_data = np.zeros((T, valid_pixels), dtype=np.float32)
        for t in range(T):
            flattened_data[t] = flat_image[t][flat_mask]
        
        # Step 8: Load events if available
        events = []
        if events_file and events_file.exists():
            events_df = bids_subject.load_events(events_file)
            events = events_df.to_dict('records')
        
        # Step 9: Create metadata
        metadata = {
            'subject_id': subject_id,
            'func_file': str(func_file.name),
            'shape': (H, W),
            'n_timepoints': T,
            'n_vertices': {
                'lh': len(lh_white_verts),
                'rh': len(rh_white_verts)
            },
            'sampling_method': self.projector.sampling_method,
            'target_size': self.target_size,
            **func_metadata
        }
        
        return {
            'bold.npy': flattened_data,  # Shape: (T, D)
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
    
    def process_subject_all_runs(self,
                               bids_subject: BIDSSubject,
                               task: Optional[str] = None,
                               run_freesurfer: bool = True) -> List[Dict]:
        """Process all functional runs for a subject.
        
        Returns:
            List of flat map data dictionaries, one per run
        """
        results = []
        
        func_files = bids_subject.get_functional_files(task=task)
        
        # Run FreeSurfer once for the subject
        if run_freesurfer and func_files:
            anat_files = bids_subject.get_anatomical_files()
            if anat_files:
                t1_file = anat_files[0]
                print(f"Running FreeSurfer for {bids_subject.subject_id}...")
                success = self.freesurfer.run_recon_all(t1_file, bids_subject.subject_id)
                if not success:
                    raise RuntimeError(f"FreeSurfer failed for {bids_subject.subject_id}")
        
        # Process each functional run
        for func_file in func_files:
            print(f"Processing {func_file.name}...")
            
            events_file = bids_subject.get_events_file(func_file)
            
            try:
                result = self.process_subject_functional(
                    bids_subject, func_file, events_file, run_freesurfer=False
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {func_file}: {e}")
                continue
        
        return results
    
    def estimate_processing_time(self, n_subjects: int, avg_runs_per_subject: int = 1) -> str:
        """Estimate total processing time."""
        # Rough estimates based on typical processing times
        freesurfer_time_hours = 6  # FreeSurfer recon-all
        projection_time_minutes = 5  # Per functional run
        
        total_freesurfer_hours = n_subjects * freesurfer_time_hours
        total_projection_hours = (n_subjects * avg_runs_per_subject * projection_time_minutes) / 60
        
        total_hours = total_freesurfer_hours + total_projection_hours
        
        return f"Estimated processing time: {total_hours:.1f} hours " \
               f"({total_freesurfer_hours:.1f}h FreeSurfer + {total_projection_hours:.1f}h projection)"