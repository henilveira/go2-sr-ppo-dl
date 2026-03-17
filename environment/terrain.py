"""
Dynamic terrain generation for curriculum learning.
Generates Perlin noise height fields with parametrized roughness levels.
"""

import numpy as np
import noise
import mujoco


def generate_perlin_heightfield(
    width: int = 128,
    height: int = 128,
    roughness_level: int = 5,  # 0-10 (0=smooth, 10=very rough)
    seed: int = 0,
    scale_factor: float = 1.0,  # 0.0-1.0 for height variation
) -> np.ndarray:
    """
    Generate Perlin noise heightfield as numpy array.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        roughness_level: 0-10 where 0 is smooth, 10 is very rough
        seed: Random seed for reproducibility
        scale_factor: Amplitude of height variations (0.0-1.0)
    
    Returns:
        2D numpy array with values in [0, 255] (grayscale)
    """
    
    # Map roughness level to Perlin parameters
    # Lower smoothness = more detail = rougher
    # roughness 0 -> smooth=100 (smooth)
    # roughness 10 -> smooth=10 (very rough)
    smooth = 100.0 - (roughness_level * 9.0)  # 100 -> 10
    
    # More octaves + higher persistence = more complexity
    octaves = 3 + int(roughness_level * 0.7)  # 3 -> 10
    persistence = 0.3 + (roughness_level * 0.05)  # 0.3 -> 0.8
    lacunarity = 2.0 + (roughness_level * 0.1)  # 2.0 -> 3.0
    
    terrain_image = np.zeros((height, width), dtype=np.uint8)
    
    np.random.seed(seed)
    random_offset_x = np.random.uniform(-1000, 1000)
    random_offset_y = np.random.uniform(-1000, 1000)
    
    for y in range(height):
        for x in range(width):
            # Add random offset to vary the seeds
            noise_value = noise.pnoise2(
                (x + random_offset_x) / smooth,
                (y + random_offset_y) / smooth,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            # Map [-1, 1] to [0, 255] with scale factor
            scaled_noise = (noise_value + 1) / 2 * scale_factor
            terrain_image[y, x] = int(np.clip(scaled_noise * 255, 0, 255))
    
    return terrain_image


def create_hfield_in_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    roughness_level: int = 5,
    hfield_idx: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate and set a height field in MuJoCo model.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        roughness_level: 0-10 roughness
        hfield_idx: Height field index in model
        seed: Random seed
    
    Returns:
        Generated height field as numpy array
    """
    
    # Generate Perlin noise heightfield
    heightfield = generate_perlin_heightfield(
        width=128,
        height=128,
        roughness_level=roughness_level,
        seed=seed,
        scale_factor=1.0,
    )
    
    # Normalize to [-1, 1] range that MuJoCo expects for hfield data
    # MuJoCo stores hfield data as normalized values
    hfield_data = (heightfield.astype(np.float32) / 255.0 - 0.5) * 2.0
    
    return hfield_data, heightfield


class TerrainCurriculum:
    """
    Manages terrain progression through curriculum learning.
    Interleaves roughness levels with incline angles.
    """
    
    def __init__(self, num_roughness_levels: int = 10):
        """
        Args:
            num_roughness_levels: Number of roughness progression stages (0-10)
        """
        self.num_roughness_levels = num_roughness_levels
        self.terrains = self._generate_curriculum()
    
    def _generate_curriculum(self) -> list:
        """
        Generate curriculum progression.
        Structure: For each roughness level, first no incline, then with incline.
        
        Returns:
            List of dicts with 'roughness', 'incline_deg', 'pitch_deg' keys
        """
        terrains = []
        
        for roughness in range(self.num_roughness_levels):
            incline_deg = ((roughness + 1) / self.num_roughness_levels) * 10.0

            # Phase 1: Roughness N with NO incline
            terrains.append({
                'name': f'rough_{roughness}_flat',
                'roughness_level': roughness,
                'pitch_deg': 0.0,
                'incline_deg': 0.0,
            })
            
            # Phase 2: Roughness N with incline
            terrains.append({
                'name': f'rough_{roughness}_inclined',
                'roughness_level': roughness,
                'pitch_deg': -incline_deg,  # Negative = tilting forward
                'incline_deg': incline_deg,
            })
        
        return terrains
    
    def get_terrain_by_progress(self, progress: float) -> dict:
        """
        Select terrain based on training progress.
        
        Args:
            progress: Training progress in [0, 1]
        
        Returns:
            Terrain configuration dict
        """
        # Map progress to terrain index
        terrain_idx = int(progress * len(self.terrains))
        terrain_idx = min(terrain_idx, len(self.terrains) - 1)
        
        return self.terrains[terrain_idx]
    
    def get_all_terrains(self) -> list:
        """Return all terrains in curriculum."""
        return self.terrains
    
    def get_terrain_count(self) -> int:
        """Return total number of terrain stages."""
        return len(self.terrains)
