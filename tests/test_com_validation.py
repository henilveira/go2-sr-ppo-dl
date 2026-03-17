"""
Valida o cálculo do centro de massa via Pinocchio contra o COM nativo do MuJoCo.

Como rodar:
    mjpython tests/test_com_validation.py              # quieto
    mjpython tests/test_com_validation.py --verbose    # imprime cada passo
    mjpython tests/test_com_validation.py --steps 200  # mais passos
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pinocchio as pin
from environment.go2_env import Go2Env


# ─────────────────────────── helpers ────────────────────────────────────────

def mujoco_com_world(env: Go2Env) -> np.ndarray:
    """COM completo do robô em frame world, calculado pelo próprio MuJoCo.

    ``data.subtree_com[body_id]`` é o COM da sub-árvore enraizada em body_id,
    expresso no frame world.  Body 'base' (id=1) cobre todo o robô.
    """
    return env.data.subtree_com[env.model.body("base").id].copy()


def pinocchio_com_world(env: Go2Env) -> np.ndarray:
    """COM completo do robô em frame world, calculado pelo Pinocchio."""
    pin_q = env._build_pinocchio_configuration()
    pin.centerOfMass(env.pin_model, env.pin_data, pin_q)
    return np.asarray(env.pin_data.com[0]).copy()


def joint_mapping_table(env: Go2Env) -> str:
    """Tabela que mostra o casamento entre env_idx → Pinocchio idx_q."""
    lines = [
        f"\n{'env_idx':>7}  {'joint_name':>26}  "
        f"{'mj_qpos_addr':>12}  {'pin_q_idx':>9}",
        "-" * 62,
    ]
    for i, name in enumerate(env.joint_names):
        lines.append(
            f"{i:>7}  {name:>26}  "
            f"{env.joint_qpos_addr[i]:>12}  {env.pin_joint_q_indices[i]:>9}"
        )
    return "\n".join(lines)


# ─────────────────────────── main ────────────────────────────────────────────

def run_validation(n_steps: int = 100, verbose: bool = False) -> bool:
    config_path = project_root / "config" / "train_config.yml"
    config = yaml.safe_load(config_path.read_text())

    env = Go2Env(config)
    obs, info = env.reset()

    # ── 1. mapeamento de juntas ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("JOINT MAPPING (env order → Pinocchio q index)")
    print("=" * 62)
    print(joint_mapping_table(env))

    # ── 2. sanidade do COM no estado inicial ─────────────────────────────────
    com_pin  = pinocchio_com_world(env)
    com_mj   = mujoco_com_world(env)
    com_base = info["center_of_mass_base"]
    diff_init = np.linalg.norm(com_pin - com_mj)

    print("\n" + "=" * 62)
    print("INITIAL STATE – COM comparison")
    print("=" * 62)
    print(f"  Pinocchio  (world) : [{com_pin[0]:+.5f}, {com_pin[1]:+.5f}, {com_pin[2]:+.5f}]")
    print(f"  MuJoCo     (world) : [{com_mj[0]:+.5f},  {com_mj[1]:+.5f},  {com_mj[2]:+.5f}]")
    print(f"  Diff (m)           : {diff_init:.6f}")
    print(f"  COM in base frame  : [{com_base[0]:+.5f}, {com_base[1]:+.5f}, {com_base[2]:+.5f}]")
    print(f"  Obs indices 30-32  : [{obs[30]:+.5f}, {obs[31]:+.5f}, {obs[32]:+.5f}]  (normalizado)")

    # ── 3. loop de passos ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"EPISODE TEST – {n_steps} random steps")
    print("=" * 62)
    if verbose:
        header = (f"{'step':>5}  {'pin_x':>9} {'pin_y':>9} {'pin_z':>9}  "
                  f"{'mj_x':>9} {'mj_y':>9} {'mj_z':>9}  {'diff_m':>9}")
        print(header)
        print("-" * len(header))

    diffs = []
    for step in range(n_steps):
        com_pin = pinocchio_com_world(env)
        com_mj  = mujoco_com_world(env)
        diff    = np.linalg.norm(com_pin - com_mj)
        diffs.append(diff)

        if verbose:
            print(
                f"{step:>5}  "
                f"{com_pin[0]:>+9.5f} {com_pin[1]:>+9.5f} {com_pin[2]:>+9.5f}  "
                f"{com_mj[0]:>+9.5f} {com_mj[1]:>+9.5f} {com_mj[2]:>+9.5f}  "
                f"{diff:>9.6f}"
            )

        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"  (episode ended early at step {step})")
            break

    env.close()

    # ── 4. estatísticas ──────────────────────────────────────────────────────
    diffs = np.asarray(diffs)
    print(f"\n  Steps ran    : {len(diffs)}")
    print(f"  Max diff (m) : {diffs.max():.6f}")
    print(f"  Mean diff (m): {diffs.mean():.6f}")
    print(f"  Std diff (m) : {diffs.std():.6f}")

    THRESHOLD_OK   = 0.005   # < 5 mm → bom
    THRESHOLD_WARN = 0.020   # < 2 cm → verificar inércias do URDF

    print()
    if diffs.max() < THRESHOLD_OK:
        print("✓  PASS – COM Pinocchio ≈ MuJoCo (< 5 mm)")
        ok = True
    elif diffs.max() < THRESHOLD_WARN:
        print("⚠  WARN – discrepância entre 5 mm e 2 cm")
        print("   Provavelmente diferença de inércias entre URDF e MJCF.")
        print("   O sinal ainda é útil, mas pode haver ruído sistemático.")
        ok = True
    else:
        print("✗  FAIL – discrepância > 2 cm")
        print("   Verifique as propriedades de inércia/massa no URDF vs MJCF.")
        ok = False

    print()
    return ok


# ─────────────────────────── entry point ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida COM Pinocchio vs MuJoCo")
    parser.add_argument("--steps",   type=int,  default=100, help="Nº de passos aleatórios")
    parser.add_argument("--verbose", action="store_true",    help="Imprime cada passo")
    args = parser.parse_args()

    passed = run_validation(n_steps=args.steps, verbose=args.verbose)
    sys.exit(0 if passed else 1)
