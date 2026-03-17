"""Generate visual previews for the dynamic terrain curriculum."""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.terrain import TerrainCurriculum, generate_perlin_heightfield


def apply_pitch_to_heightfield(heightfield: np.ndarray, pitch_deg: float) -> np.ndarray:
    """Apply a simple tilt along x-axis to visualize inclined terrain."""
    if abs(pitch_deg) < 1e-6:
        return heightfield

    h, w = heightfield.shape
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    tilt = np.tan(np.deg2rad(abs(pitch_deg))) * x
    tilt = np.tile(tilt[None, :], (h, 1))

    # Keep visualization scale comparable by re-normalizing after adding tilt.
    tilted = heightfield.astype(np.float32) + tilt * 255.0 * 0.25
    tilted = tilted - tilted.min()
    max_val = tilted.max() if tilted.max() > 0 else 1.0
    tilted = (tilted / max_val) * 255.0
    return tilted.astype(np.uint8)


def save_curriculum_preview(output_path: Path, seed: int = 42) -> None:
    """Render all 20 phases (10 roughness x flat/inclined) as a grid image."""
    curriculum = TerrainCurriculum(num_roughness_levels=10)
    terrains = curriculum.get_all_terrains()

    fig, axes = plt.subplots(4, 5, figsize=(20, 14), constrained_layout=True)
    axes = axes.flatten()

    for idx, terrain in enumerate(terrains):
        roughness = terrain["roughness_level"]
        pitch_deg = terrain["pitch_deg"]

        base = generate_perlin_heightfield(
            width=64,
            height=64,
            roughness_level=roughness,
            seed=seed + roughness,
            scale_factor=1.0,
        )
        preview = apply_pitch_to_heightfield(base, pitch_deg)

        ax = axes[idx]
        im = ax.imshow(preview, cmap="terrain", origin="lower")
        ax.set_title(
            f"P{idx:02d} | rough={roughness} | pitch={pitch_deg:.0f}deg",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, shrink=0.5)
    cbar.set_label("Relative height")

    fig.suptitle("Dynamic Terrain Curriculum Preview (20 phases)", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_roughness_progress_plot(output_path: Path, seed: int = 42) -> None:
    """Plot roughness progression using std as a quick proxy of irregularity."""
    roughness_levels = list(range(10))
    stds = []

    for roughness in roughness_levels:
        hfield = generate_perlin_heightfield(
            width=64,
            height=64,
            roughness_level=roughness,
            seed=seed + roughness,
            scale_factor=1.0,
        )
        stds.append(float(np.std(hfield)))

    plt.figure(figsize=(8, 4))
    plt.plot(roughness_levels, stds, marker="o")
    plt.title("Roughness Progression (height std proxy)")
    plt.xlabel("Roughness level")
    plt.ylabel("Std of height values")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "logs" / "terrain_preview"

    preview_path = output_dir / "terrain_curriculum_preview.png"
    progress_path = output_dir / "roughness_progress.png"

    save_curriculum_preview(preview_path)
    save_roughness_progress_plot(progress_path)

    print("Saved terrain previews:")
    print(f" - {preview_path}")
    print(f" - {progress_path}")


if __name__ == "__main__":
    main()
