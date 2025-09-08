"""Surface flattening utilities for converting 3D surfaces to 2D flat maps."""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import cv2


class SurfaceFlattener:
    """Converts 3D cortical surfaces to 2D flat map representations."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        
    def flatten_surface_data(self, vertices_3d: np.ndarray, faces: np.ndarray, 
                           vertices_flat: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map surface vertex data to 2D flat image coordinates.
        
        Args:
            vertices_3d: 3D surface vertices (N, 3)
            faces: Surface faces (M, 3) 
            vertices_flat: 2D flat coordinates (N, 2)
            data: Data at vertices (N, T) for T timepoints
            
        Returns:
            flat_image: 2D image representation (T, H, W)
            mask: Binary mask indicating valid regions (H, W)
        """
        H, W = self.target_size
        
        # Normalize flat coordinates to image space
        flat_coords = self._normalize_flat_coordinates(vertices_flat, (H, W))
        
        # Create 2D grid
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        grid_points = np.column_stack([y_grid.ravel(), x_grid.ravel()])
        
        # Build KDTree for efficient nearest neighbor search
        tree = cKDTree(flat_coords)
        
        # For each grid point, find nearest surface vertex
        distances, indices = tree.query(grid_points, k=1)
        
        # Create mask based on distance threshold
        max_dist = np.sqrt(2)  # Allow up to sqrt(2) pixel distance
        valid_mask = distances <= max_dist
        
        # Initialize output
        T = data.shape[1] if len(data.shape) > 1 else 1
        flat_image = np.zeros((T, H, W))
        mask = np.zeros((H, W), dtype=bool)
        
        # Map data to grid
        valid_indices = indices[valid_mask]
        valid_grid_idx = np.where(valid_mask)[0]
        
        for i, grid_idx in enumerate(valid_grid_idx):
            y, x = divmod(grid_idx, W)
            vertex_idx = valid_indices[i]
            
            if len(data.shape) > 1:
                flat_image[:, y, x] = data[vertex_idx, :]
            else:
                flat_image[0, y, x] = data[vertex_idx]
            mask[y, x] = True
        
        # Apply morphological operations to clean up mask
        mask = self._clean_mask(mask)
        
        return flat_image.squeeze() if T == 1 else flat_image, mask
    
    def _normalize_flat_coordinates(self, flat_coords: np.ndarray, 
                                  target_size: Tuple[int, int]) -> np.ndarray:
        """Normalize flat coordinates to target image size."""
        H, W = target_size
        
        # Remove invalid coordinates (e.g., flat surface boundary artifacts)
        valid_idx = np.isfinite(flat_coords).all(axis=1)
        coords = flat_coords[valid_idx]
        
        if len(coords) == 0:
            return np.zeros((0, 2))
        
        # Normalize to [0, 1] range
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        coord_range = max_coords - min_coords
        
        # Handle degenerate cases
        coord_range[coord_range == 0] = 1
        
        normalized = (coords - min_coords) / coord_range
        
        # Scale to target size with padding
        padding = 0.05  # 5% padding
        scale_factor = np.array([H, W]) * (1 - 2 * padding)
        offset = np.array([H, W]) * padding
        
        scaled_coords = normalized * scale_factor + offset
        
        # Create full array with invalid coordinates set to [-1, -1]
        result = np.full((len(flat_coords), 2), -1.0)
        result[valid_idx] = scaled_coords
        
        return result
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up mask using morphological operations."""
        # Convert to uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Remove small disconnected components
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        return (mask_uint8 > 0).astype(bool)
    
    def create_hemisphere_flat_map(self, lh_vertices_3d: np.ndarray, lh_faces: np.ndarray,
                                 lh_vertices_flat: np.ndarray, lh_data: np.ndarray,
                                 rh_vertices_3d: np.ndarray, rh_faces: np.ndarray, 
                                 rh_vertices_flat: np.ndarray, rh_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create combined flat map for both hemispheres.
        
        Returns:
            flat_image: Combined flat image (T, H, W)
            mask: Combined mask (H, W)
        """
        H, W = self.target_size
        
        # Process each hemisphere separately
        lh_image, lh_mask = self.flatten_surface_data(
            lh_vertices_3d, lh_faces, lh_vertices_flat, lh_data
        )
        rh_image, rh_mask = self.flatten_surface_data(
            rh_vertices_3d, rh_faces, rh_vertices_flat, rh_data
        )
        
        # Ensure proper dimensions
        if len(lh_image.shape) == 2:
            lh_image = lh_image[None]
        if len(rh_image.shape) == 2:
            rh_image = rh_image[None]
            
        T = max(lh_image.shape[0], rh_image.shape[0])
        
        # Combine hemispheres side by side
        combined_image = np.zeros((T, H, W))
        combined_mask = np.zeros((H, W), dtype=bool)
        
        # Left hemisphere on left half, right hemisphere on right half
        W_half = W // 2
        
        # Resize hemispheres to fit in half-width
        lh_resized = self._resize_hemisphere_data(lh_image, lh_mask, (H, W_half))
        rh_resized = self._resize_hemisphere_data(rh_image, rh_mask, (H, W_half))
        
        # Place hemispheres
        combined_image[:, :, :W_half] = lh_resized[0]
        combined_image[:, :, W_half:] = rh_resized[0]
        combined_mask[:, :W_half] = lh_resized[1]
        combined_mask[:, W_half:] = rh_resized[1]
        
        return combined_image, combined_mask
    
    def _resize_hemisphere_data(self, image: np.ndarray, mask: np.ndarray, 
                              target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Resize hemisphere data to target size."""
        H_target, W_target = target_size
        T, H_orig, W_orig = image.shape
        
        if H_orig == H_target and W_orig == W_target:
            return image, mask
        
        # Resize using OpenCV
        resized_image = np.zeros((T, H_target, W_target))
        for t in range(T):
            resized_image[t] = cv2.resize(
                image[t].astype(np.float32), 
                (W_target, H_target), 
                interpolation=cv2.INTER_LINEAR
            )
        
        resized_mask = cv2.resize(
            mask.astype(np.uint8), 
            (W_target, H_target), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        return resized_image, resized_mask
    
    def create_sparse_mask(self, mask: np.ndarray) -> coo_matrix:
        """Convert dense mask to sparse COO format for efficient storage."""
        H, W = mask.shape
        y_coords, x_coords = np.where(mask)
        data = np.ones(len(y_coords), dtype=np.uint8)
        
        return coo_matrix((data, (y_coords, x_coords)), shape=(H, W))