#!/usr/bin/env python3
"""
Main script for converting BIDS fMRI data to flat map WebDatasets.

Usage:
    python process_bids_to_flatmaps.py --bids_dir /path/to/bids --output_dir /path/to/output
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bids.loader import BIDSLoader
from flat.generator import FlatMapGenerator  
from webdataset.packager import WebDatasetPackager


def main():
    parser = argparse.ArgumentParser(
        description="Convert BIDS fMRI data to flat map WebDatasets"
    )
    
    # Required arguments
    parser.add_argument(
        "--bids_dir", 
        required=True,
        help="Path to BIDS dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        required=True, 
        help="Output directory for WebDataset shards"
    )
    parser.add_argument(
        "--subjects_dir",
        required=True,
        help="FreeSurfer subjects directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--freesurfer_home",
        help="FreeSurfer installation directory"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Specific subjects to process (default: all)"
    )
    parser.add_argument(
        "--task",
        help="Specific task to process (default: all tasks)"
    )
    parser.add_argument(
        "--target_size",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("HEIGHT", "WIDTH"),
        help="Target size for flat maps (default: 256 256)"
    )
    parser.add_argument(
        "--sampling_method",
        choices=["ribbon", "nearest", "trilinear"],
        default="ribbon", 
        help="Volume to surface sampling method (default: ribbon)"
    )
    parser.add_argument(
        "--max_samples_per_shard",
        type=int,
        default=100,
        help="Maximum samples per WebDataset shard (default: 100)"
    )
    parser.add_argument(
        "--dataset_name",
        default="flat",
        help="Base name for dataset shards (default: flat)"
    )
    parser.add_argument(
        "--skip_freesurfer",
        action="store_true",
        help="Skip FreeSurfer reconstruction (assumes already done)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be processed without actually processing"
    )
    parser.add_argument(
        "--disable_detailed_logging",
        action="store_true",
        help="Disable detailed step-by-step logging"
    )
    parser.add_argument(
        "--log_dir",
        help="Directory where per-subject FreeSurfer and pipeline logs will be saved"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    bids_dir = Path(args.bids_dir)
    if not bids_dir.exists():
        print(f"Error: BIDS directory does not exist: {bids_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    subjects_dir = Path(args.subjects_dir)
    
    # Initialize components
    print("Initializing BIDS loader...")
    bids_loader = BIDSLoader(bids_dir)
    
    print("Initializing flat map generator...")
    generator = FlatMapGenerator(
        subjects_dir=subjects_dir,
        target_size=tuple(args.target_size),
        sampling_method=args.sampling_method,
        freesurfer_home=args.freesurfer_home,
        enable_detailed_logging=not args.disable_detailed_logging,
        log_dir=args.log_dir
    )
    
    # Load historical timing data if available
    timing_file = output_dir / "processing_times.json"
    if timing_file.exists():
        print(f"Loading historical timing data from {timing_file}")
        generator.load_historical_times(timing_file)
    else:
        print("No historical timing data found - will use default estimates")
    
    print("Initializing WebDataset packager...")
    packager = WebDatasetPackager(
        output_dir=output_dir,
        max_samples_per_shard=args.max_samples_per_shard
    )
    
    # Get subjects to process
    all_subjects = bids_loader.get_subjects()
    if args.subjects:
        subjects_to_process = [s for s in args.subjects if s in all_subjects]
        if not subjects_to_process:
            print(f"Error: None of the specified subjects found in dataset")
            print(f"Available subjects: {all_subjects}")
            sys.exit(1)
    else:
        subjects_to_process = all_subjects
    
    print(f"Found {len(all_subjects)} subjects in BIDS dataset")
    print(f"Will process {len(subjects_to_process)} subjects")
    
    if args.task:
        available_tasks = bids_loader.get_tasks()
        if args.task not in available_tasks:
            print(f"Error: Task '{args.task}' not found in dataset")
            print(f"Available tasks: {available_tasks}")
            sys.exit(1)
        print(f"Processing task: {args.task}")
    else:
        print("Processing all available tasks")
    
    # Estimate processing time
    total_runs = 0
    for subject_id in subjects_to_process:
        subject = bids_loader.get_subject(subject_id)
        func_files = subject.get_functional_files(task=args.task)
        total_runs += len(func_files)
    
    avg_runs_per_subject = total_runs / len(subjects_to_process) if subjects_to_process else 0
    time_estimate = generator.estimate_processing_time(len(subjects_to_process), avg_runs_per_subject)
    print(f"\\n{time_estimate}")
    print(f"Total functional runs to process: {total_runs}")
    
    if args.dry_run:
        print("\\nDry run mode - showing what would be processed:")
        for subject_id in subjects_to_process:
            subject = bids_loader.get_subject(subject_id)
            func_files = subject.get_functional_files(task=args.task)
            print(f"  {subject_id}: {len(func_files)} functional runs")
        return
    
    # Confirm before starting
    if not args.skip_freesurfer:
        response = input("\\nFreeSurfer reconstruction will be run. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled by user")
            return
    
    # Process subjects with progress tracking
    all_samples = []
    
    # Get progress bar configuration
    progress_config = generator.get_progress_bar_config(len(subjects_to_process), avg_runs_per_subject)
    
    # Calculate total functional runs for progress tracking
    total_runs_processed = 0
    with tqdm(**progress_config) as pbar:
        for i, subject_id in enumerate(subjects_to_process):
            if not args.disable_detailed_logging:
                print(f"\\n{'='*80}")
                print(f"SUBJECT {i+1}/{len(subjects_to_process)}: {subject_id}")
                print(f"{'='*80}")
            else:
                pbar.set_description(f"Processing {subject_id}")
            
            try:
                subject = bids_loader.get_subject(subject_id)
                samples = generator.process_subject_all_runs(
                    subject, 
                    task=args.task,
                    run_freesurfer=not args.skip_freesurfer
                )
                
                all_samples.extend(samples)
                runs_in_subject = len(samples)
                total_runs_processed += runs_in_subject
                
                # Update progress bar by the number of runs processed
                pbar.update(runs_in_subject)
                pbar.set_postfix({
                    'subject': subject_id,
                    'runs': runs_in_subject,
                    'total_samples': len(all_samples)
                })
                
                if not args.disable_detailed_logging:
                    print(f"\\n✅ Subject {subject_id} completed: {runs_in_subject} functional runs processed")
                
            except Exception as e:
                print(f"❌ Error processing {subject_id}: {e}")
                # Still need to update progress bar even on error
                subject_obj = bids_loader.get_subject(subject_id)
                expected_runs = len(subject_obj.get_functional_files(task=args.task))
                pbar.update(expected_runs)
                continue
    
    # Package into WebDataset
    if all_samples:
        print(f"\\n{'='*80}")
        print(f"PACKAGING {len(all_samples)} samples into WebDataset")
        print(f"{'='*80}")
        
        # Show progress for packaging as well
        with tqdm(total=len(all_samples), desc="Packaging samples", unit="sample") as pbar:
            def progress_callback(completed):
                pbar.update(completed - pbar.n)
            
            shard_files = packager.package_samples(all_samples, args.dataset_name, progress_callback)
        
        # Validate shards with progress
        print("\\nValidating created shards...")
        with tqdm(shard_files, desc="Validating shards", unit="shard") as pbar:
            for shard_file in pbar:
                pbar.set_postfix({'file': shard_file.name})
                packager.validate_shard(shard_file)
        
        # Print statistics
        stats = packager.get_dataset_stats(shard_files)
        print(f"\\n=== Dataset Statistics ===")
        print(f"Shards created: {stats['n_shards']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"Average samples per shard: {stats['avg_samples_per_shard']}")
        print(f"Average size per shard: {stats['avg_mb_per_shard']:.1f} MB")
        
        print(f"\\nDataset created successfully in: {output_dir}")
        print("\\nTo use with flat_data.py, use:")
        print(f"  url = '{output_dir}/*.tar'")
        
    else:
        print("\\nNo samples generated. Check for errors above.")
    
    # Save updated timing data for future runs
    if not args.disable_detailed_logging:
        print(f"\\nSaving timing data for future estimates...")
        generator.save_historical_times(timing_file)
        
        # Show processing statistics if we have data
        stats = generator.get_processing_statistics()
        if 'message' not in stats:
            print(f"\\n{'='*60}")
            print("PROCESSING STATISTICS")
            print(f"{'='*60}")
            
            for step, step_stats in stats.items():
                if step != 'summary' and isinstance(step_stats, dict):
                    print(f"{step:<25} {step_stats['mean_time']:>8.2f}s ± {step_stats['std_time']:>6.2f}s (n={step_stats['n_samples']})")
            
            summary = stats['summary']
            print(f"\\nMost time-consuming step: {summary['most_time_consuming_step']}")
            print(f"Average total time per run: {summary['total_mean_time_per_run']:.2f}s")
            print(f"Historical sessions: {summary['total_processing_sessions']}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
