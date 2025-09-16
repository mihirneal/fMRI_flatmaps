#!/usr/bin/env python3
"""
Test script for processing a single subject through the fMRI flatmaps pipeline.
This helps verify the pipeline works before scaling to multiple subjects.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bids.loader import BIDSLoader
from flat.generator import FlatMapGenerator
from webdataset.packager import WebDatasetPackager


def test_single_subject(subject_id: str):
    """Test processing pipeline on a single subject."""
    
    # Configuration
    bids_dir = Path("/teamspace/studios/this_studio/ampa_ds")
    subjects_dir = Path("/teamspace/studios/this_studio/freesurfer_subjects")  
    output_dir = Path("/teamspace/studios/this_studio/ampa_test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Testing pipeline with subject: {subject_id}")
    print(f"BIDS directory: {bids_dir}")
    print(f"FreeSurfer subjects directory: {subjects_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize components
    print("\n=== Initializing components ===")
    bids_loader = BIDSLoader(bids_dir)
    
    generator = FlatMapGenerator(
        subjects_dir=subjects_dir,
        target_size=(256, 256),
        sampling_method='ribbon',
        enable_detailed_logging=True  # Enable detailed logging for debugging
    )
    
    packager = WebDatasetPackager(
        output_dir=output_dir,
        max_samples_per_shard=100
    )
    
    # Check if subject exists
    available_subjects = bids_loader.get_subjects()
    if subject_id not in available_subjects:
        print(f"ERROR: Subject {subject_id} not found!")
        print(f"Available subjects: {available_subjects}")
        return False
    
    print(f"Subject {subject_id} found in BIDS dataset")
    
    # Get subject info
    subject = bids_loader.get_subject(subject_id)
    func_files = subject.get_functional_files()
    anat_files = subject.get_anatomical_files()
    
    print(f"Anatomical files: {len(anat_files)}")
    print(f"Functional files: {len(func_files)}")
    
    if not anat_files:
        print("ERROR: No anatomical files found!")
        return False
    
    if not func_files:
        print("ERROR: No functional files found!")
        return False
    
    # Show what we'll process
    print(f"\nWill process:")
    print(f"  T1w file: {anat_files[0].name}")
    for i, func_file in enumerate(func_files):
        print(f"  Functional run {i+1}: {func_file.name}")
    
    # Auto-confirm for testing
    print(f"\nProceeding with processing {subject_id}...")
    
    # Process the subject
    try:
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBJECT: {subject_id}")
        print(f"{'='*80}")
        
        samples = generator.process_subject_all_runs(
            subject, 
            task=None,  # Process all tasks
            run_freesurfer=True
        )
        
        if not samples:
            print("ERROR: No samples generated!")
            return False
            
        print(f"\nSuccessfully generated {len(samples)} samples")
        
        # Package into WebDataset
        print(f"\n=== Packaging samples ===")
        shard_files = packager.package_samples(samples, f"test_{subject_id}")
        
        # Validate
        print(f"\n=== Validating shards ===")
        for shard_file in shard_files:
            packager.validate_shard(shard_file)
        
        # Print results
        stats = packager.get_dataset_stats(shard_files)
        print(f"\n{'='*60}")
        print(f"SUCCESS! Dataset created for {subject_id}")
        print(f"{'='*60}")
        print(f"Shards created: {stats['n_shards']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"Output location: {output_dir}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_single_subject.py <subject_id>")
        print("Example: python test_single_subject.py sub-ASD01")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    success = test_single_subject(subject_id)
    sys.exit(0 if success else 1)