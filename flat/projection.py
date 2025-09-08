"""Volume to surface projection for fMRI data."""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial import cKDTree
import nibabel as nib
from nibabel import freesurfer


class VolumeToSurfaceProjector:
    """Projects fMRI volume data to cortical surface vertices."""
    
    def __init__(self, sampling_method: str = 'ribbon', n_samples: int = 3):
        """
        Args:
            sampling_method: 'ribbon', 'nearest', or 'trilinear'
            n_samples: Number of samples along cortical ribbon (for ribbon method)
        """
        self.sampling_method = sampling_method
        self.n_samples = n_samples
    
    def project_to_surface(self, volume_data: np.ndarray, volume_affine: np.ndarray,
                          white_vertices: np.ndarray, pial_vertices: np.ndarray) -> np.ndarray:
        """Project volume data to surface vertices.
        
        Args:
            volume_data: fMRI volume data (X, Y, Z, T)
            volume_affine: Affine transform from voxel to world coordinates
            white_vertices: White matter surface vertices (N, 3)
            pial_vertices: Pial surface vertices (N, 3)
            
        Returns:
            surface_data: Data at surface vertices (N, T)
        """
        X, Y, Z = volume_data.shape[:3]
        T = volume_data.shape[3] if len(volume_data.shape) > 3 else 1
        N = len(white_vertices)
        
        if self.sampling_method == 'ribbon':
            surface_data = self._ribbon_sampling(
                volume_data, volume_affine, white_vertices, pial_vertices
            )
        elif self.sampling_method == 'nearest':
            surface_data = self._nearest_neighbor_sampling(
                volume_data, volume_affine, (white_vertices + pial_vertices) / 2
            )
        elif self.sampling_method == 'trilinear':
            surface_data = self._trilinear_sampling(
                volume_data, volume_affine, (white_vertices + pial_vertices) / 2
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        return surface_data
    
    def _ribbon_sampling(self, volume_data: np.ndarray, volume_affine: np.ndarray,
                        white_vertices: np.ndarray, pial_vertices: np.ndarray) -> np.ndarray:
        """Sample along cortical ribbon between white matter and pial surfaces."""
        N = len(white_vertices)
        T = volume_data.shape[3] if len(volume_data.shape) > 3 else 1
        
        # Create sampling points along ribbon
        sample_fractions = np.linspace(0, 1, self.n_samples)
        surface_data = np.zeros((N, T))
        
        # Inverse affine to go from world to voxel coordinates
        inv_affine = np.linalg.inv(volume_affine)
        
        for i in range(N):
            white_point = white_vertices[i]
            pial_point = pial_vertices[i]
            
            # Sample along ribbon
            ribbon_samples = []
            for frac in sample_fractions:
                world_point = white_point + frac * (pial_point - white_point)
                
                # Convert to voxel coordinates
                voxel_point = self._world_to_voxel(world_point, inv_affine)
                
                # Sample volume data
                sample = self._sample_volume_at_point(volume_data, voxel_point)
                if sample is not None:
                    ribbon_samples.append(sample)
            
            # Average samples along ribbon
            if ribbon_samples:
                surface_data[i] = np.mean(ribbon_samples, axis=0)
        
        return surface_data
    
    def _nearest_neighbor_sampling(self, volume_data: np.ndarray, volume_affine: np.ndarray,
                                 surface_vertices: np.ndarray) -> np.ndarray:
        """Sample using nearest neighbor interpolation."""
        N = len(surface_vertices)
        T = volume_data.shape[3] if len(volume_data.shape) > 3 else 1
        surface_data = np.zeros((N, T))
        
        inv_affine = np.linalg.inv(volume_affine)
        
        for i, vertex in enumerate(surface_vertices):
            voxel_point = self._world_to_voxel(vertex, inv_affine)
            sample = self._sample_volume_at_point(volume_data, voxel_point, method='nearest')
            if sample is not None:
                surface_data[i] = sample
        
        return surface_data
    
    def _trilinear_sampling(self, volume_data: np.ndarray, volume_affine: np.ndarray,
                          surface_vertices: np.ndarray) -> np.ndarray:
        """Sample using trilinear interpolation."""
        N = len(surface_vertices)
        T = volume_data.shape[3] if len(volume_data.shape) > 3 else 1
        surface_data = np.zeros((N, T))
        
        inv_affine = np.linalg.inv(volume_affine)
        
        for i, vertex in enumerate(surface_vertices):
            voxel_point = self._world_to_voxel(vertex, inv_affine)
            sample = self._sample_volume_at_point(volume_data, voxel_point, method='trilinear')
            if sample is not None:
                surface_data[i] = sample
        
        return surface_data
    
    def _world_to_voxel(self, world_point: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel coordinates."""
        # Add homogeneous coordinate
        world_homo = np.append(world_point, 1)
        voxel_homo = inv_affine @ world_homo
        return voxel_homo[:3]
    
    def _sample_volume_at_point(self, volume_data: np.ndarray, voxel_point: np.ndarray, 
                              method: str = 'nearest') -> Optional[np.ndarray]:
        """Sample volume data at a given voxel coordinate."""
        X, Y, Z = volume_data.shape[:3]
        x, y, z = voxel_point
        
        # Check bounds
        if not (0 <= x < X and 0 <= y < Y and 0 <= z < Z):
            return None
        
        if method == 'nearest':
            ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
            ix = np.clip(ix, 0, X-1)
            iy = np.clip(iy, 0, Y-1)  
            iz = np.clip(iz, 0, Z-1)
            
            if len(volume_data.shape) == 4:
                return volume_data[ix, iy, iz, :]
            else:
                return np.array([volume_data[ix, iy, iz]])
        
        elif method == 'trilinear':
            # Trilinear interpolation
            x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
            x1, y1, z1 = min(x0 + 1, X-1), min(y0 + 1, Y-1), min(z0 + 1, Z-1)
            
            # Interpolation weights
            wx = x - x0
            wy = y - y0
            wz = z - z0
            
            # Sample at 8 corners
            if len(volume_data.shape) == 4:
                c000 = volume_data[x0, y0, z0, :] * (1-wx) * (1-wy) * (1-wz)
                c001 = volume_data[x0, y0, z1, :] * (1-wx) * (1-wy) * wz
                c010 = volume_data[x0, y1, z0, :] * (1-wx) * wy * (1-wz)
                c011 = volume_data[x0, y1, z1, :] * (1-wx) * wy * wz
                c100 = volume_data[x1, y0, z0, :] * wx * (1-wy) * (1-wz)
                c101 = volume_data[x1, y0, z1, :] * wx * (1-wy) * wz
                c110 = volume_data[x1, y1, z0, :] * wx * wy * (1-wz)
                c111 = volume_data[x1, y1, z1, :] * wx * wy * wz
            else:
                c000 = volume_data[x0, y0, z0] * (1-wx) * (1-wy) * (1-wz)
                c001 = volume_data[x0, y0, z1] * (1-wx) * (1-wy) * wz
                c010 = volume_data[x0, y1, z0] * (1-wx) * wy * (1-wz)
                c011 = volume_data[x0, y1, z1] * (1-wx) * wy * wz
                c100 = volume_data[x1, y0, z0] * wx * (1-wy) * (1-wz)
                c101 = volume_data[x1, y0, z1] * wx * (1-wy) * wz
                c110 = volume_data[x1, y1, z0] * wx * wy * (1-wz)
                c111 = volume_data[x1, y1, z1] * wx * wy * wz
                
                result = c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111
                return np.array([result])
            
            return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111
        
        return None