"""Convert data/demos.npz into a Minari dataset.

Minari is Farama Foundation's offline-RL dataset hub. Once converted,
the dataset can be loaded by anyone with:

    import minari
    ds = minari.load_dataset("airhockey-sac-v0")

This script:
  1. Loads our raw .npz demonstrations
  2. Splits them into per-episode trajectories
  3. Wraps each one in a Minari EpisodeBuffer
  4. Creates a local Minari dataset
  5. Optionally pushes it to the public Minari registry via the CLI

Run:
    python scripts/build_minari_dataset.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build(npz_path: str, dataset_id: str, out_dir: str) -> None:
    try:
        import minari
        from minari import DataCollector, MinariDataset
    except ImportError:
        raise SystemExit(
            "Minari is not installed. Install with: pip install minari"
        )

    import gymnasium as gym
    from airhockey.env import AirHockeyEnv

    d = np.load(npz_path)
    obs = d["obs"]
    act = d["act"]
    episode = d["episode"]

    n_episodes = int(episode.max()) + 1
    print(f"Loaded {len(obs):,} transitions across {n_episodes:,} episodes")

    # Wrap our env in DataCollector and replay each demonstration through it.
    env = AirHockeyEnv()
    env = DataCollector(env, record_infos=False)

    for ep in range(n_episodes):
        mask = episode == ep
        ep_obs = obs[mask]
        ep_act = act[mask]
        env.reset(seed=ep)
        for a in ep_act:
            env.step(a.astype(np.float32))

    print("Creating Minari dataset…")
    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="SoftActorCritic",
        author="akshan-main",
        author_email="",
        code_permalink="https://github.com/akshan-main/airhockey-rl",
    )

    print(f"\n✓ Dataset created locally as {dataset_id}")
    print(f"  Episodes: {dataset.total_episodes}")
    print(f"  Steps: {dataset.total_steps}")
    print("\nTo push to the public Minari registry:")
    print(f"  minari upload {dataset_id}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/demos.npz")
    p.add_argument("--id", type=str, default="airhockey-sac-v0")
    p.add_argument("--out", type=str, default="minari_export")
    args = p.parse_args()
    build(args.data, args.id, args.out)


if __name__ == "__main__":
    main()
