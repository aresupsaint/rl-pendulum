"""Evaluate a saved TD3 actor on fresh, deterministic test seeds.

Example:
  python evaluate_td3.py results_td3/InvertedPendulum-v5/seed0/best.pt
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from td3_inverted_pendulums import aggregate_evaluation, make_mlp, pendulum_upright


@torch.no_grad()
def run(checkpoint, episodes, seed):
    # weights_only=True accepts only tensors and simple metadata.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    env_name = saved["environment"]
    env = gym.make(env_name)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    actor = make_mlp(obs_dim, act_dim)
    actor.load_state_dict(saved["actor"])
    actor.eval()
    low = np.asarray(saved["low"], dtype=np.float32)
    high = np.asarray(saved["high"], dtype=np.float32)
    middle = (low + high) / 2
    scale = (high - low) / 2
    rows = []
    try:
        for index in range(episodes):
            obs, _ = env.reset(seed=seed + index)
            total = 0.0
            length = 0
            upright = []
            while True:
                state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                normalized = actor(state).tanh().numpy()[0]
                action = np.clip(middle + scale * normalized, low, high).astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                if env_name == "Pendulum-v1":
                    upright.append(pendulum_upright(obs))
                if terminated or truncated:
                    if env_name == "Pendulum-v1":
                        success = np.mean(upright[-50:]) >= 0.8
                        upright_fraction = float(np.mean(upright))
                    else:
                        success = bool(truncated and not terminated)
                        upright_fraction = float("nan")
                    rows.append({"return": total, "length": length,
                                 "success": bool(success),
                                 "upright_fraction": upright_fraction})
                    break
    finally:
        env.close()
    result = aggregate_evaluation(rows, int(saved["env_steps"]))
    result.update({"checkpoint": str(checkpoint.resolve()),
                   "environment": env_name, "test_seed": seed,
                   "test_episodes": episodes})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=300_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    result = run(args.checkpoint, args.episodes, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
