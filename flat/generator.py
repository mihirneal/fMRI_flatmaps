"""Flat map generation combining surface processing and projection."""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import time
import logging
import re
from tqdm import tqdm

from bids.loader import BIDSSubject
from surface.freesurfer import FreeSurferProcessor
from surface.flattening import SurfaceFlattener
from flat.projection import VolumeToSurfaceProjector


class ProcessingLogger:
    """Handles step-by-step logging and timing for fMRI flatmap processing."""
    
    def __init__(self, logger_name: str = 'fMRI_flatmaps'):
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False

        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self.formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.step_times = {}
        self.current_step = None
        self.step_start_time = None
        self.file_handler: Optional[logging.Handler] = None
        self.current_log_path: Optional[Path] = None

    def configure_run(self, log_dir: Path, run_label: str):
        """Attach a file handler for the current run and reset timing state."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        safe_label = re.sub(r'[^A-Za-z0-9._-]+', '_', run_label)
        log_path = log_dir / f"{safe_label}.log"

        if self.file_handler is not None:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None

        self.file_handler = logging.FileHandler(log_path, mode='w')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        self.logger.info(f"Writing detailed logs to {log_path}")

        self.current_log_path = log_path
        self.step_times = {}
        self.current_step = None
        self.step_start_time = None

    def close_run(self):
        """Detach the file handler once a run completes."""
        if self.file_handler is not None:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None
        self.current_log_path = None
    
    def start_step(self, step_name: str, description: str = ""):
        """Start timing a processing step."""
        if self.current_step:
            self.end_step()
        
        self.current_step = step_name
        self.step_start_time = time.time()
        
        msg = f"Starting step: {step_name}"
        if description:
            msg += f" - {description}"
        self.logger.info(msg)
    
    def end_step(self):
        """End timing the current processing step."""
        if self.current_step and self.step_start_time:
            elapsed = time.time() - self.step_start_time
            self.step_times[self.current_step] = elapsed
            self.logger.info(f"Completed step: {self.current_step} ({elapsed:.2f}s)")
            self.current_step = None
            self.step_start_time = None
    
    def log_substep(self, message: str):
        """Log a substep within the current step."""
        self.logger.info(f"  └─ {message}")
    
    def get_step_times(self) -> Dict[str, float]:
        """Get dictionary of step names to elapsed times."""
        return self.step_times.copy()
    
    def log_summary(self):
        """Log summary of all step times."""
        if not self.step_times:
            return
        
        total_time = sum(self.step_times.values())
        self.logger.info("\n" + "="*60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("="*60)
        
        for step, time_taken in self.step_times.items():
            percentage = (time_taken / total_time) * 100
            self.logger.info(f"{step:<30} {time_taken:>8.2f}s ({percentage:>5.1f}%)")
        
        self.logger.info("-"*60)
        self.logger.info(f"{'TOTAL TIME':<30} {total_time:>8.2f}s ({100.0:>5.1f}%)")
        self.logger.info("="*60)


class FlatMapGenerator:
    """Generates flat maps from BIDS fMRI data."""
    
    def __init__(self, 
                 subjects_dir: str | Path,
                 target_size: Tuple[int, int] = (256, 256),
                 sampling_method: str = 'ribbon',
                 freesurfer_home: Optional[str] = None,
                 enable_detailed_logging: bool = True,
                 log_dir: Optional[str | Path] = None):
        
        self.subjects_dir = Path(subjects_dir)
        self.freesurfer = FreeSurferProcessor(self.subjects_dir, freesurfer_home)
        self.flattener = SurfaceFlattener(target_size)
        self.projector = VolumeToSurfaceProjector(sampling_method)
        self.target_size = target_size
        self.enable_detailed_logging = enable_detailed_logging
        
        self.log_root_dir = Path(log_dir) if log_dir else None
        if self.log_root_dir:
            self.log_root_dir.mkdir(parents=True, exist_ok=True)

        if enable_detailed_logging:
            self.logger = ProcessingLogger()
        
        # Store historical timing data for better estimates
        self.historical_times = {
            'freesurfer_recon': [],
            'load_functional': [],
            'load_surfaces': [], 
            'project_to_surface': [],
            'create_flat_maps': [],
            'create_sparse_mask': [],
            'flatten_data': [],
            'load_events': []
        }
    
    def _get_subject_log_dir(self, subject_id: str) -> Path:
        """Determine and create the log directory for a subject."""
        if self.log_root_dir is not None:
            log_dir = self.log_root_dir / subject_id
        else:
            log_dir = self.subjects_dir / subject_id / "flatmaps_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def process_subject_functional(self, 
                                 bids_subject: BIDSSubject,
                                 func_file: Path,
                                 events_file: Optional[Path] = None,
                                 run_freesurfer: bool = True,
                                 subject_log_dir: Optional[Path] = None) -> Dict:
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
        
        subject_log_dir = Path(subject_log_dir) if subject_log_dir else self._get_subject_log_dir(subject_id)

        if self.enable_detailed_logging:
            self.logger.configure_run(subject_log_dir, func_file.stem)
            self.logger.logger.info(f"\n{'='*80}")
            self.logger.logger.info(f"PROCESSING: {subject_id} - {func_file.name}")
            self.logger.logger.info(f"{'='*80}")

        # Step 1: Surface reconstruction (if needed)
        if run_freesurfer:
            if self.enable_detailed_logging:
                self.logger.start_step("freesurfer_recon", f"Running FreeSurfer recon-all for {subject_id}")
            
            anat_files = bids_subject.get_anatomical_files()
            if not anat_files:
                raise ValueError(f"No anatomical files found for {subject_id}")
            
            # Use first T1w file
            t1_file = anat_files[0]
            if self.enable_detailed_logging:
                self.logger.log_substep(f"Using T1 file: {t1_file.name}")
                
            success = self.freesurfer.run_recon_all(t1_file, subject_id, log_dir=subject_log_dir)
            if not success:
                raise RuntimeError(f"FreeSurfer failed for {subject_id}")
                
            if self.enable_detailed_logging:
                self.logger.end_step()
        
        # Step 2: Load functional data
        if self.enable_detailed_logging:
            self.logger.start_step("load_functional", f"Loading functional data: {func_file.name}")
            
        func_data, func_header, func_metadata = bids_subject.load_functional_data(func_file)
        func_affine = func_header.get_best_affine()
        
        if self.enable_detailed_logging:
            self.logger.log_substep(f"Data shape: {func_data.shape}")
            self.logger.log_substep(f"Affine matrix loaded")
            self.logger.end_step()
        
        # Step 3: Load surface data
        if self.enable_detailed_logging:
            self.logger.start_step("load_surfaces", f"Loading surface data for {subject_id}")
            
        try:
            # White matter surfaces
            if self.enable_detailed_logging:
                self.logger.log_substep("Loading left hemisphere white surface")
            lh_white_verts, lh_faces = self.freesurfer.load_surface(subject_id, 'lh', 'white')
            
            if self.enable_detailed_logging:
                self.logger.log_substep("Loading right hemisphere white surface")
            rh_white_verts, rh_faces = self.freesurfer.load_surface(subject_id, 'rh', 'white')
            
            # Pial surfaces  
            if self.enable_detailed_logging:
                self.logger.log_substep("Loading left hemisphere pial surface")
            lh_pial_verts, _ = self.freesurfer.load_surface(subject_id, 'lh', 'pial')
            
            if self.enable_detailed_logging:
                self.logger.log_substep("Loading right hemisphere pial surface")
            rh_pial_verts, _ = self.freesurfer.load_surface(subject_id, 'rh', 'pial')
            
            # Flat surfaces
            if self.enable_detailed_logging:
                self.logger.log_substep("Loading left hemisphere flat surface")
            lh_flat_verts, _ = self.freesurfer.load_flat_surface(subject_id, 'lh', log_dir=subject_log_dir)
            
            if self.enable_detailed_logging:
                self.logger.log_substep("Loading right hemisphere flat surface")
            rh_flat_verts, _ = self.freesurfer.load_flat_surface(subject_id, 'rh', log_dir=subject_log_dir)
            
        except FileNotFoundError as e:
            raise RuntimeError(f"Surface files missing for {subject_id}: {e}")
        
        if self.enable_detailed_logging:
            self.logger.log_substep(f"LH vertices: {len(lh_white_verts)}, RH vertices: {len(rh_white_verts)}")
            self.logger.end_step()
        
        # Step 4: Project functional data to surfaces
        if self.enable_detailed_logging:
            self.logger.start_step("project_to_surface", f"Projecting functional data to surfaces")
        
        lh_surface_data = self.projector.project_to_surface(
            func_data, func_affine, lh_white_verts, lh_pial_verts
        )
        
        if self.enable_detailed_logging:
            self.logger.log_substep(f"LH projection complete: {lh_surface_data.shape}")
            
        rh_surface_data = self.projector.project_to_surface(
            func_data, func_affine, rh_white_verts, rh_pial_verts
        )
        
        if self.enable_detailed_logging:
            self.logger.log_substep(f"RH projection complete: {rh_surface_data.shape}")
            self.logger.end_step()
        
        # Step 5: Create flat maps
        if self.enable_detailed_logging:
            self.logger.start_step("create_flat_maps", f"Generating flat maps")
        
        flat_image, flat_mask = self.flattener.create_hemisphere_flat_map(
            lh_white_verts, lh_faces, lh_flat_verts, lh_surface_data,
            rh_white_verts, rh_faces, rh_flat_verts, rh_surface_data
        )
        
        if self.enable_detailed_logging:
            self.logger.log_substep(f"Flat image shape: {flat_image.shape}")
            self.logger.log_substep(f"Valid pixels: {flat_mask.sum()}/{flat_mask.size}")
            self.logger.end_step()
        
        # Step 6: Create sparse mask
        if self.enable_detailed_logging:
            self.logger.start_step("create_sparse_mask", "Creating sparse mask")
            
        sparse_mask = self.flattener.create_sparse_mask(flat_mask)
        
        if self.enable_detailed_logging:
            self.logger.log_substep(f"Sparse mask nnz: {sparse_mask.nnz}")
            self.logger.end_step()
        
        # Step 7: Flatten image data for storage
        if self.enable_detailed_logging:
            self.logger.start_step("flatten_data", "Flattening image data for storage")
            
        # flat_image shape: (T, H, W), we want to store as (T, D) where D = number of valid pixels
        T, H, W = flat_image.shape
        valid_pixels = flat_mask.sum()
        
        # Extract data only at valid mask locations
        flattened_data = np.zeros((T, valid_pixels), dtype=np.float32)
        for t in range(T):
            flattened_data[t] = flat_image[t][flat_mask]
        
        if self.enable_detailed_logging:
            self.logger.log_substep(f"Flattened data shape: {flattened_data.shape}")
            self.logger.log_substep(f"Memory usage: {flattened_data.nbytes / 1024 / 1024:.1f} MB")
            self.logger.end_step()
        
        # Step 8: Load events if available
        if self.enable_detailed_logging:
            self.logger.start_step("load_events", "Loading events file")
            
        events = []
        if events_file and events_file.exists():
            events_df = bids_subject.load_events(events_file)
            events = events_df.to_dict('records')
            if self.enable_detailed_logging:
                self.logger.log_substep(f"Loaded {len(events)} events from {events_file.name}")
        else:
            if self.enable_detailed_logging:
                self.logger.log_substep("No events file found")
                
        if self.enable_detailed_logging:
            self.logger.end_step()
        
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
        
        # Log processing summary for this run
        try:
            if self.enable_detailed_logging:
                self.logger.log_summary()
                
                # Store timing data for future estimates
                step_times = self.logger.get_step_times()
                for step, time_taken in step_times.items():
                    if step in self.historical_times:
                        self.historical_times[step].append(time_taken)
        finally:
            if self.enable_detailed_logging:
                self.logger.close_run()

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

        subject_id = bids_subject.subject_id
        subject_log_dir = self._get_subject_log_dir(subject_id)
        
        func_files = bids_subject.get_functional_files(task=task)
        
        # Run FreeSurfer once for the subject
        if run_freesurfer and func_files:
            anat_files = bids_subject.get_anatomical_files()
            if anat_files:
                t1_file = anat_files[0]
                print(f"Running FreeSurfer for {bids_subject.subject_id}...")
                success = self.freesurfer.run_recon_all(t1_file, bids_subject.subject_id, log_dir=subject_log_dir)
                if not success:
                    raise RuntimeError(f"FreeSurfer failed for {bids_subject.subject_id}")
        
        # Process each functional run
        for func_file in func_files:
            print(f"Processing {func_file.name}...")
            
            events_file = bids_subject.get_events_file(func_file)
            
            try:
                result = self.process_subject_functional(
                    bids_subject, func_file, events_file, run_freesurfer=False, subject_log_dir=subject_log_dir
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {func_file}: {e}")
                continue
        
        return results
    
    def estimate_processing_time(self, n_subjects: int, avg_runs_per_subject: int = 1) -> str:
        """Estimate total processing time using historical data when available."""
        # Use historical timing data if available, otherwise fall back to rough estimates
        if self.historical_times['freesurfer_recon']:
            avg_freesurfer_time = np.mean(self.historical_times['freesurfer_recon'])
        else:
            avg_freesurfer_time = 6 * 3600  # 6 hours in seconds (default estimate)
        
        # Calculate average time for processing a single run (excluding FreeSurfer)
        run_processing_time = 0
        for step in ['load_functional', 'load_surfaces', 'project_to_surface', 
                     'create_flat_maps', 'create_sparse_mask', 'flatten_data', 'load_events']:
            if self.historical_times[step]:
                run_processing_time += np.mean(self.historical_times[step])
            else:
                # Default estimates in seconds
                defaults = {
                    'load_functional': 30,
                    'load_surfaces': 60, 
                    'project_to_surface': 240,
                    'create_flat_maps': 120,
                    'create_sparse_mask': 10,
                    'flatten_data': 30,
                    'load_events': 5
                }
                run_processing_time += defaults[step]
        
        total_freesurfer_time = n_subjects * avg_freesurfer_time
        total_run_processing_time = n_subjects * avg_runs_per_subject * run_processing_time
        total_time = total_freesurfer_time + total_run_processing_time
        
        # Convert to hours for display
        total_hours = total_time / 3600
        freesurfer_hours = total_freesurfer_time / 3600
        run_processing_hours = total_run_processing_time / 3600
        
        historical_note = " (using historical data)" if any(self.historical_times.values()) else " (using default estimates)"
        
        return f"Estimated processing time: {total_hours:.1f} hours{historical_note}\n" \
               f"  ├─ FreeSurfer reconstruction: {freesurfer_hours:.1f}h ({freesurfer_hours/total_hours*100:.1f}%)\n" \
               f"  └─ Functional processing: {run_processing_hours:.1f}h ({run_processing_hours/total_hours*100:.1f}%)"
    
    def get_progress_bar_config(self, n_subjects: int, avg_runs_per_subject: int = 1) -> Dict:
        """Get configuration for tqdm progress bar with accurate time estimates."""
        total_runs = n_subjects * avg_runs_per_subject
        
        # Estimate time per run (excluding FreeSurfer since it runs once per subject)
        run_processing_time = 0
        for step in ['load_functional', 'load_surfaces', 'project_to_surface', 
                     'create_flat_maps', 'create_sparse_mask', 'flatten_data', 'load_events']:
            if self.historical_times[step]:
                run_processing_time += np.mean(self.historical_times[step])
            else:
                defaults = {
                    'load_functional': 30, 'load_surfaces': 60, 'project_to_surface': 240,
                    'create_flat_maps': 120, 'create_sparse_mask': 10, 'flatten_data': 30, 'load_events': 5
                }
                run_processing_time += defaults[step]
        
        return {
            'total': total_runs,
            'desc': 'Processing fMRI runs',
            'unit': 'run',
            'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        }
    
    def save_historical_times(self, filepath: str | Path):
        """Save historical timing data to JSON file for future use."""
        import json
        
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.historical_times, f, indent=2)
    
    def load_historical_times(self, filepath: str | Path):
        """Load historical timing data from JSON file."""
        import json
        
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, 'r') as f:
                loaded_times = json.load(f)
                
            # Merge with existing data
            for step, times in loaded_times.items():
                if step in self.historical_times:
                    self.historical_times[step].extend(times)
                else:
                    self.historical_times[step] = times
    
    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics."""
        if not any(self.historical_times.values()):
            return {'message': 'No historical timing data available'}
        
        stats = {}
        total_samples = 0
        
        for step, times in self.historical_times.items():
            if times:
                stats[step] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'median_time': np.median(times),
                    'n_samples': len(times)
                }
                total_samples = max(total_samples, len(times))
        
        stats['summary'] = {
            'total_processing_sessions': total_samples,
            'most_time_consuming_step': max(stats.keys(), 
                                          key=lambda k: stats[k]['mean_time'] if isinstance(stats[k], dict) else 0),
            'total_mean_time_per_run': sum(s['mean_time'] for s in stats.values() if isinstance(s, dict))
        }
        
        return stats
