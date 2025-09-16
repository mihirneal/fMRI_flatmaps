"""FreeSurfer-based surface processing."""

import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque
import shlex
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

    def _build_clean_env(self) -> dict:
        """Build a clean environment for invoking FreeSurfer binaries.

        Strips potentially conflicting variables coming from Python/venv/OpenCV
        (e.g., LD_LIBRARY_PATH, QT_QPA_PLATFORM_PLUGIN_PATH) that can cause
        FreeSurfer tools (mri_synthmorph, etc.) to abort.
        """
        env = {}

        # Preserve minimal sane defaults
        env['HOME'] = os.environ.get('HOME', str(Path.home()))
        env['USER'] = os.environ.get('USER', '')
        env['LOGNAME'] = os.environ.get('LOGNAME', env['USER'])
        env['LANG'] = os.environ.get('LANG', 'C.UTF-8')
        env['LC_ALL'] = os.environ.get('LC_ALL', 'C.UTF-8')

        # FreeSurfer core vars
        if 'FREESURFER_HOME' in os.environ:
            fs_home = os.environ['FREESURFER_HOME']
            env['FREESURFER_HOME'] = fs_home
            env['FREESURFER'] = fs_home
            env['MNI_DIR'] = f"{fs_home}/mni"
            env['FSL_DIR'] = f"{fs_home}/fsl"
            # PATH: put FS bin first, then system
            env['PATH'] = f"{fs_home}/bin:/usr/local/bin:/usr/bin:/bin"
        else:
            # Fall back to system path
            env['PATH'] = os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')

        # Subjects location
        env['SUBJECTS_DIR'] = str(self.subjects_dir)
        env['FUNCTIONALS_DIR'] = str(self.subjects_dir)

        # Threading controls to keep tools stable/consistent
        env['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
        env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = os.environ.get(
            'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '1'
        )

        # Do NOT propagate these (source of many segfaults/aborts)
        blocked_prefixes = (
            'PYTHON',        # PYTHONPATH, PYTHONHOME, etc.
            'VIRTUAL_ENV',
            'CONDA',
            'QT_',           # Qt plugin/font dirs from cv2
        )
        blocked_exact = {
            'LD_LIBRARY_PATH',
            'LD_PRELOAD',
            'DYLD_LIBRARY_PATH',
        }
        for k, v in os.environ.items():
            if k in env:
                continue
            if k in blocked_exact:
                continue
            if any(k.startswith(pfx) for pfx in blocked_prefixes):
                continue
            # Allow other innocuous vars (eg TMPDIR, SSL_CERT_*, etc.)
            env[k] = v

        return env
    
    def _resolve_log_path(self, subject_id: str, log_dir: Optional[Path], filename: str) -> Path:
        """Resolve the path where command logs should be written."""
        if log_dir is not None:
            base_dir = Path(log_dir)
        else:
            base_dir = self.subjects_dir / subject_id / "flatmaps_logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / filename

    def _run_with_logging(self, cmd: List[str], log_path: Path, env: dict) -> Tuple[int, List[str]]:
        """Run a command, stream output to log, and return exit code and tail."""
        tail_buffer: deque[str] = deque(maxlen=50)
        command_str = " ".join(shlex.quote(str(part)) for part in cmd)
        start_ts = datetime.utcnow().isoformat() + 'Z'

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w', encoding='utf-8') as log_file:
            log_file.write(f"# Command: {command_str}\n")
            log_file.write(f"# Started: {start_ts}\n\n")
            log_file.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )

            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                tail_buffer.append(line.rstrip())

            process.stdout.close()
            returncode = process.wait()

            end_ts = datetime.utcnow().isoformat() + 'Z'
            log_file.write(f"\n# Finished: {end_ts}\n")
            log_file.write(f"# Exit code: {returncode}\n")

        return returncode, list(tail_buffer)

    def run_recon_all(self, t1_file: Path, subject_id: str, 
                     stages: Optional[List[str]] = None, force: bool = False,
                     log_dir: Optional[Path] = None) -> bool:
        """Run FreeSurfer recon-all pipeline."""
        if stages is None:
            stages = ['-all']  # Full pipeline
            
        # Check if already processed
        subject_dir = self.subjects_dir / subject_id
        if not force and (subject_dir / 'surf' / 'lh.white').exists():
            print(f"Subject {subject_id} already processed, skipping...")
            return True
            
        # Use full path to recon-all if available
        recon_all_cmd = 'recon-all'
        if 'FREESURFER_HOME' in os.environ:
            recon_all_path = Path(os.environ['FREESURFER_HOME']) / 'bin' / 'recon-all'
            if recon_all_path.exists():
                recon_all_cmd = str(recon_all_path)
        else:
            # Try common FreeSurfer locations
            common_paths = ['/usr/local/freesurfer/8.1.0/bin/recon-all', 
                           '/usr/local/freesurfer/bin/recon-all',
                           '/opt/freesurfer/bin/recon-all']
            for path in common_paths:
                if Path(path).exists():
                    recon_all_cmd = path
                    break
        
        cmd = [
            recon_all_cmd,
            '-subject', subject_id,
            '-i', str(t1_file),
            '-sd', str(self.subjects_dir)
        ] + stages

        # If user requested multi-threading via env, add recon-all flags
        try:
            n_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
        except ValueError:
            n_threads = 1
        if n_threads > 1:
            cmd += ['-parallel', '-openmp', str(n_threads)]
        
        log_path = self._resolve_log_path(subject_id, log_dir, 'recon-all.log')

        print(f"Running FreeSurfer recon-all for {subject_id} (log: {log_path})")
        
        # Build clean environment for FreeSurfer
        env = self._build_clean_env()
        
        try:
            returncode, tail = self._run_with_logging(cmd, log_path, env)
        except OSError as e:
            print(f"Failed to start FreeSurfer for {subject_id}: {e}")
            with log_path.open('a', encoding='utf-8') as log_file:
                log_file.write(f"\n# Failed to launch command: {e}\n")
            return False

        if returncode == 0:
            print(f"FreeSurfer completed successfully for {subject_id}. Log saved to {log_path}")
            return True

        print(f"FreeSurfer failed for {subject_id}. See log: {log_path}")
        if tail:
            print("Last log lines:")
            for line in tail[-10:]:
                print(line)

        # If core surfaces exist, allow pipeline to proceed
        required_surfaces = [
            self.subjects_dir / subject_id / 'surf' / 'lh.white',
            self.subjects_dir / subject_id / 'surf' / 'rh.white',
            self.subjects_dir / subject_id / 'surf' / 'lh.pial',
            self.subjects_dir / subject_id / 'surf' / 'rh.pial',
            self.subjects_dir / subject_id / 'surf' / 'lh.inflated',
            self.subjects_dir / subject_id / 'surf' / 'rh.inflated',
        ]
        if all(p.exists() for p in required_surfaces):
            print("FreeSurfer reported an error, but required surface files are present. Proceeding.")
            return True

        return False
    
    def load_surface(self, subject_id: str, hemi: str, surface: str = 'white') -> Tuple[np.ndarray, np.ndarray]:
        """Load surface vertices and faces."""
        surf_file = self.subjects_dir / subject_id / 'surf' / f'{hemi}.{surface}'
        if not surf_file.exists():
            raise FileNotFoundError(f"Surface file not found: {surf_file}")
            
        vertices, faces = freesurfer.read_geometry(surf_file)
        return vertices, faces
    
    def load_flat_surface(self, subject_id: str, hemi: str, log_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load flattened surface coordinates."""
        flat_file = self.subjects_dir / subject_id / 'surf' / f'{hemi}.flat'
        if not flat_file.exists():
            # Try to create flat surface
            self.create_flat_surface(subject_id, hemi, log_dir=log_dir)
            
        vertices, faces = freesurfer.read_geometry(flat_file)
        return vertices, faces
    
    def create_flat_surface(self, subject_id: str, hemi: str, log_dir: Optional[Path] = None) -> bool:
        """Create flattened surface using mris_flatten."""
        cmd = [
            'mris_flatten',
            '-w', '0',  # No smoothing
            str(self.subjects_dir / subject_id / 'surf' / f'{hemi}.inflated'),
            str(self.subjects_dir / subject_id / 'surf' / f'{hemi}.flat')
        ]
        
        log_path = self._resolve_log_path(subject_id, log_dir, f'{hemi}.mris_flatten.log')

        try:
            returncode, tail = self._run_with_logging(cmd, log_path, self._build_clean_env())
        except OSError as e:
            print(f"mris_flatten failed to start for {subject_id} ({hemi}): {e}")
            with log_path.open('a', encoding='utf-8') as log_file:
                log_file.write(f"\n# Failed to launch command: {e}\n")
            return False

        if returncode == 0:
            return True

        print(f"mris_flatten failed for {subject_id} ({hemi}). See log: {log_path}")
        if tail:
            print("Last log lines:")
            for line in tail[-10:]:
                print(line)
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
