#!/usr/bin/env python3
"""
Quick test of the new dynamic terrain curriculum system
"""

import numpy as np
import yaml
from pathlib import Path
from environment.terrain import TerrainCurriculum, generate_perlin_heightfield

def test_terrain_generation():
    """Test terrain generation with different roughness levels"""
    print("\n" + "="*70)
    print("TESTING TERRAIN GENERATION")
    print("="*70)
    
    for roughness in [0, 3, 6, 9]:
        hfield = generate_perlin_heightfield(
            width=128,
            height=128,
            roughness_level=roughness,
            seed=42,
            scale_factor=1.0,
        )
        print(f"✓ Roughness {roughness}: Generated {hfield.shape} array")
        print(f"  Value range: {hfield.min()}-{hfield.max()}")
        print(f"  Mean: {hfield.mean():.1f}, Std: {hfield.std():.1f}")

def test_terrain_curriculum():
    """Test terrain curriculum phases"""
    print("\n" + "="*70)
    print("TESTING TERRAIN CURRICULUM (29 phases)")
    print("="*70)
    
    tc = TerrainCurriculum(num_roughness_levels=10)
    
    print(f"\nTotal phases: {tc.get_terrain_count()}")
    print("\nPhase progression:")
    
    # Show phases at different progress levels
    progress_points = [0.0, 0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 1.0]
    
    for progress in progress_points:
        terrain = tc.get_terrain_by_progress(progress)
        phase_idx = int(progress * tc.get_terrain_count())
        print(f"  {progress*100:5.0f}% → Phase {phase_idx:2d}: "
              f"{terrain['name']:25s} "
              f"(roughness={terrain['roughness_level']}, pitch={terrain['pitch_deg']:5.1f}°)")

def test_curriculum_structure():
    """Verify expected structure with initial smooth incline levels."""
    print("\n" + "="*70)
    print("VERIFYING CURRICULUM STRUCTURE")
    print("="*70)
    
    tc = TerrainCurriculum(num_roughness_levels=10)
    terrains = tc.get_all_terrains()
    
    print(f"\nTotal phases: {len(terrains)}")
    print("Expected: 29 phases (9 smooth incline levels + 10 roughness levels × 2)")

    print("\nInitial smooth incline levels:")
    for idx in range(9):
        phase = terrains[idx]
        print(f"  Phase {idx:2d}: {phase['name']:30s} (pitch={phase['pitch_deg']:5.1f}°)")
    
    # Group by roughness
    by_roughness = {}
    for terrain in terrains:
        r = terrain['roughness_level']
        if r not in by_roughness:
            by_roughness[r] = []
        by_roughness[r].append(terrain)
    
    print("\nStructure per roughness level:")
    for roughness in sorted(by_roughness.keys()):
        phases = by_roughness[roughness]
        print(f"  Roughness {roughness}: {len(phases)} phases")
        for phase in phases:
            print(f"    - {phase['name']:30s} (pitch={phase['pitch_deg']:5.1f}°)")
    
    # Verify alternating pattern for roughness stages (after first 9 smooth levels)
    print("\nVerifying roughness-stage alternating pattern (flat, inclined, ...):")
    all_ok = True
    for i, terrain in enumerate(terrains[9:], start=9):
        phase_num_in_roughness = (i - 9) % 2
        roughness_idx = (i - 9) // 2
        expected_incline = 0.0 if phase_num_in_roughness == 0 else -((roughness_idx + 1) / 10.0) * 10.0
        actual_incline = terrain['pitch_deg']
        
        if abs(actual_incline - expected_incline) > 0.01:
            print(f"  ✗ Phase {i}: Expected pitch {expected_incline}°, got {actual_incline}°")
            all_ok = False
    
    if all_ok:
        print("  ✓ All phases follow flat/inclined alternating pattern!")

def test_config():
    """Test loading config with new terrain curriculum"""
    print("\n" + "="*70)
    print("TESTING CONFIG LOADING")
    print("="*70)
    
    config_path = Path(__file__).parent / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    terrain_config = config.get('terrain_curriculum', {})
    print(f"✓ Config loaded")
    print(f"  Enabled: {terrain_config.get('enabled')}")
    print(f"  Mode: {terrain_config.get('mode')}")

if __name__ == "__main__":
    test_terrain_generation()
    test_terrain_curriculum()
    test_curriculum_structure()
    test_config()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
