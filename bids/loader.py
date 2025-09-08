"""BIDS dataset loader for fMRI data."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import nibabel as nib
import numpy as np


class BIDSSubject:
    """Represents a single subject in a BIDS dataset."""
    
    def __init__(self, subject_dir: Path):
        self.subject_dir = Path(subject_dir)
        self.subject_id = self.subject_dir.name
        
    def get_anatomical_files(self) -> List[Path]:
        """Get all anatomical (T1w) files for this subject."""
        anat_dir = self.subject_dir / "anat"
        if not anat_dir.exists():
            return []
        return list(anat_dir.glob("*_T1w.nii*"))
    
    def get_functional_files(self, task: Optional[str] = None) -> List[Path]:
        """Get all functional files for this subject, optionally filtered by task."""
        func_dir = self.subject_dir / "func"
        if not func_dir.exists():
            return []
            
        pattern = "*_bold.nii*"
        if task:
            pattern = f"*_task-{task}_*_bold.nii*"
            
        return list(func_dir.glob(pattern))
    
    def load_functional_data(self, func_file: Path) -> Tuple[np.ndarray, nib.Nifti1Header, Dict]:
        """Load functional data and metadata."""
        # Load NIfTI file
        img = nib.load(func_file)
        data = img.get_fdata()
        
        # Load sidecar JSON if available
        json_file = func_file.with_suffix('.json')
        metadata = {}
        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                
        return data, img.header, metadata
    
    def load_anatomical_data(self, anat_file: Path) -> Tuple[np.ndarray, nib.Nifti1Header]:
        """Load anatomical data."""
        img = nib.load(anat_file)
        return img.get_fdata(), img.header
    
    def get_events_file(self, func_file: Path) -> Optional[Path]:
        """Get events file corresponding to functional file."""
        # Replace _bold.nii* with _events.tsv
        events_file = func_file.parent / func_file.name.replace('_bold.nii', '_events.tsv')
        if events_file.exists():
            return events_file
        return None
    
    def load_events(self, events_file: Path) -> pd.DataFrame:
        """Load events from TSV file."""
        return pd.read_csv(events_file, sep='\t')


class BIDSLoader:
    """BIDS dataset loader."""
    
    def __init__(self, bids_root: str | Path):
        self.bids_root = Path(bids_root)
        
        # Load dataset description
        desc_file = self.bids_root / "dataset_description.json"
        if desc_file.exists():
            with open(desc_file, 'r') as f:
                self.dataset_description = json.load(f)
        else:
            self.dataset_description = {}
            
        # Load participants info if available
        participants_file = self.bids_root / "participants.tsv"
        if participants_file.exists():
            self.participants = pd.read_csv(participants_file, sep='\t')
        else:
            self.participants = None
    
    def get_subjects(self) -> List[str]:
        """Get list of subject IDs in the dataset."""
        subject_dirs = [d for d in self.bids_root.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')]
        return [d.name for d in subject_dirs]
    
    def get_subject(self, subject_id: str) -> BIDSSubject:
        """Get BIDSSubject instance for given subject ID."""
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'
        
        subject_dir = self.bids_root / subject_id
        if not subject_dir.exists():
            raise ValueError(f"Subject {subject_id} not found in {self.bids_root}")
            
        return BIDSSubject(subject_dir)
    
    def get_tasks(self) -> List[str]:
        """Get list of unique task names in the dataset."""
        tasks = set()
        for subject_id in self.get_subjects():
            subject = self.get_subject(subject_id)
            func_files = subject.get_functional_files()
            for func_file in func_files:
                # Extract task name from filename
                parts = func_file.name.split('_')
                for part in parts:
                    if part.startswith('task-'):
                        task_name = part[5:]  # Remove 'task-' prefix
                        tasks.add(task_name)
        return list(tasks)
    
    def iter_functional_data(self, task: Optional[str] = None):
        """Iterator over all functional data in the dataset."""
        for subject_id in self.get_subjects():
            subject = self.get_subject(subject_id)
            func_files = subject.get_functional_files(task=task)
            
            for func_file in func_files:
                events_file = subject.get_events_file(func_file)
                yield {
                    'subject_id': subject_id,
                    'func_file': func_file,
                    'events_file': events_file,
                    'subject': subject
                }