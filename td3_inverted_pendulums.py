"""Train TD3 on one- and two-link inverted pendulum balance tasks.

Examples:
  python td3_inverted_pendulums.py --env InvertedPendulum-v5 --steps 200000
  python td3_inverted_pendulums.py --env InvertedDoublePendulum-v5 --steps 500000
  python td3_inverted_pendulums.py --env Pendulum-v1 --steps 200000

Pendulum-v1 is a single-link swing-up task. The MuJoCo environments are
upright-balance tasks. All three have continuous Box action spaces.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ENVIRONMENTS = (
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Pendulum-v1",
)
TRAIN_FIELDS = ("episode", "env_steps", "return", "length", "terminated", "truncated")
EVAL_FIELDS = ("env_steps", "mean_return", "std_return", "mean_length",
               "success_rate", "upright_fraction")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, output_dim),
    )


class ReplayBuffer:
    """Store normalized actions so Q functions always see actions in [-1, 1]."""

    def __init__(self, capacity, obs_dim, act_dim, seed):
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, act_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.position
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.next_obs[i] = next_obs
        self.terminated[i, 0] = terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return tuple(torch.as_tensor(array[idx], device=device)
                     for array in (self.obs, self.actions, self.rewards,
                                   self.next_obs, self.terminated))

    def state_dict(self):
        return {"obs": self.obs, "actions": self.actions,
                "rewards": self.rewards, "next_obs": self.next_obs,
                "terminated": self.terminated, "size": self.size,
                "position": self.position,
                "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        for name in ("obs", "actions", "rewards", "next_obs", "terminated"):
            value = state[name]
            if value.shape != getattr(self, name).shape:
                raise ValueError(f"Replay buffer shape mismatch for {name}")
            getattr(self, name)[:] = value
        self.size = int(state["size"])
        self.position = int(state["position"])
        self.rng.bit_generator.state = state["rng_state"]


class TD3:
    def __init__(self, obs_dim, low, high, device, args, seed):
        self.device = device
        self.low = np.asarray(low, dtype=np.float32).reshape(-1)
        self.high = np.asarray(high, dtype=np.float32).reshape(-1)
        self.mid = (self.high + self.low) / 2
        self.scale = (self.high - self.low) / 2
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("TD3 needs finite action bounds with high > low")
        act_dim = len(self.low)
        self.actor = make_mlp(obs_dim, act_dim).to(device)
        self.actor_target = make_mlp(obs_dim, act_dim).to(device)
        self.q1 = make_mlp(obs_dim + act_dim, 1).to(device)
        self.q2 = make_mlp(obs_dim + act_dim, 1).to(device)
        self.q1_target = make_mlp(obs_dim + act_dim, 1).to(device)
        self.q2_target = make_mlp(obs_dim + act_dim, 1).to(device)
        for source, target in ((self.actor, self.actor_target),
                               (self.q1, self.q1_target),
                               (self.q2, self.q2_target)):
            target.load_state_dict(source.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.critic_lr)
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.exploration_noise = args.exploration_noise
        self.policy_delay = args.policy_delay
        self.updates = 0
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def select_normalized(self, obs, explore):
        state = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device).unsqueeze(0)
        action = self.actor(state).tanh().cpu().numpy()[0]
        if explore:
            action += self.rng.normal(0, self.exploration_noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def to_environment_action(self, normalized):
        return np.clip(self.mid + self.scale * normalized,
                       self.low, self.high).astype(np.float32)

    @staticmethod
    @torch.no_grad()
    def soft_update(source, target, tau):
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.lerp_(src, tau)

    def update(self, batch):
        obs, action, reward, next_obs, terminated = batch
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs).tanh() + noise).clamp(-1, 1)
            next_input = torch.cat((next_obs, next_action), dim=1)
            next_q = torch.minimum(self.q1_target(next_input),
                                   self.q2_target(next_input))
            # Truncation is a time limit, so only true termination stops bootstrapping.
            target = reward + self.gamma * (1 - terminated) * next_q

        critic_input = torch.cat((obs, action), dim=1)
        critic_loss = (F.mse_loss(self.q1(critic_input), target)
                       + F.mse_loss(self.q2(critic_input), target))
        self.q_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_opt.step()
        self.updates += 1

        if self.updates % self.policy_delay == 0:
            for p in list(self.q1.parameters()) + list(self.q2.parameters()):
                p.requires_grad_(False)
            actor_action = self.actor(obs).tanh()
            actor_loss = -self.q1(torch.cat((obs, actor_action), dim=1)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            for p in list(self.q1.parameters()) + list(self.q2.parameters()):
                p.requires_grad_(True)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.q1, self.q1_target, self.tau)
            self.soft_update(self.q2, self.q2_target, self.tau)

    def save_policy(self, path, env_name, seed, steps):
        torch.save({"actor": self.actor.state_dict(), "environment": env_name,
                    "seed": seed, "env_steps": steps, "low": self.low.tolist(),
                    "high": self.high.tolist()}, path)

    def training_state(self):
        return {"actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "q_opt": self.q_opt.state_dict(), "updates": self.updates,
                "rng_state": self.rng.bit_generator.state}

    def load_training_state(self, state):
        for name in ("actor", "actor_target", "q1", "q2", "q1_target", "q2_target"):
            getattr(self, name).load_state_dict(state[name])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.q_opt.load_state_dict(state["q_opt"])
        self.updates = int(state["updates"])
        self.rng.bit_generator.state = state["rng_state"]


def pendulum_upright(obs):
    """An interpretable metric, separate from the environment's reward."""
    return bool(obs[0] >= np.cos(np.deg2rad(15)) and abs(obs[2]) < 0.5)


def evaluate(agent, env_name, seed, episodes):
    env = gym.make(env_name)
    rows = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            length = 0
            upright = []
            while True:
                normalized = agent.select_normalized(obs, explore=False)
                action = agent.to_environment_action(normalized)
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
                        # For the MuJoCo balance tasks, lasting to the time limit is success.
                        success = bool(truncated and not terminated)
                        upright_fraction = float("nan")
                    rows.append({"return": total, "length": length,
                                 "success": bool(success),
                                 "upright_fraction": upright_fraction})
                    break
    finally:
        env.close()
    return rows


def aggregate_evaluation(rows, env_steps):
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    upright = [row["upright_fraction"] for row in rows]
    sem = float(returns.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0
    return {"env_steps": env_steps,
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std(ddof=1)) if len(rows) > 1 else 0.0,
            "return_ci95_low": float(returns.mean() - 1.96 * sem),
            "return_ci95_high": float(returns.mean() + 1.96 * sem),
            "median_return": float(np.median(returns)),
            "mean_length": float(np.mean([row["length"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "upright_fraction": (float(np.mean(upright))
                                 if np.all(np.isfinite(upright)) else None)}


def mean_ci95(values):
    """Student-t confidence interval for a small number of training seeds."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return [mean, mean]
    critical_by_n = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                     6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
                     10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
                     14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120,
                     18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
                     22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
                     26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048,
                     30: 2.045}
    critical = critical_by_n.get(len(values), 1.96)
    margin = critical * float(values.std(ddof=1)) / np.sqrt(len(values))
    return [mean - margin, mean + margin]


def evaluate_baseline(env_name, controller, seed, episodes):
    """Evaluate random and swing-up/PD baselines on Pendulum-v1."""
    if controller == "pd" and env_name != "Pendulum-v1":
        raise ValueError("The PD baseline is currently defined only for Pendulum-v1")
    env = gym.make(env_name)
    rows = []
    try:
        for index in range(episodes):
            env.action_space.seed(seed + index)
            obs, _ = env.reset(seed=seed + index)
            total, upright = 0.0, []
            while True:
                if controller == "random":
                    action = env.action_space.sample()
                else:
                    theta = float(np.arctan2(obs[1], obs[0]))
                    omega = float(obs[2])
                    # Energy shaping swings up from arbitrary angles; PD damps
                    # the pendulum after it enters the upright neighborhood.
                    energy = 0.5 * omega * omega + 15.0 * np.cos(theta)
                    target_energy = 15.0
                    if abs(theta) < 0.35 and abs(omega) < 2.0:
                        torque = -12.0 * theta - 3.0 * omega
                    else:
                        direction = np.sign(omega)
                        if direction == 0:
                            direction = np.sign(-theta) or 1.0
                        torque = 0.25 * (target_energy - energy) * direction
                    action = np.asarray([np.clip(torque, env.action_space.low[0],
                                                 env.action_space.high[0])], dtype=np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                upright.append(pendulum_upright(obs) if env_name == "Pendulum-v1" else False)
                if terminated or truncated:
                    rows.append({"return": total, "length": len(upright),
                                 "success": bool(np.mean(upright[-50:]) >= 0.8)
                                 if env_name == "Pendulum-v1" else bool(truncated and not terminated),
                                 "upright_fraction": float(np.mean(upright))
                                 if env_name == "Pendulum-v1" else float("nan")})
                    break
    finally:
        env.close()
    result = aggregate_evaluation(rows, 0)
    result["controller"] = controller
    result["episodes"] = episodes
    return result, rows


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_resume_checkpoint(path, *, env_name, seed, step, episode, obs,
                          agent, replay, train_rows, eval_rows, best_return,
                          next_evaluation, next_checkpoint, action_space):
    """Save a resumable snapshot at an episode boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"version": 1, "environment": env_name, "seed": seed,
                "step": step, "episode": episode, "obs": np.asarray(obs),
                "agent": agent.training_state(), "replay": replay.state_dict(),
                "train_rows": train_rows, "eval_rows": eval_rows,
                "best_return": best_return,
                "next_evaluation": next_evaluation,
                "next_checkpoint": next_checkpoint,
                "action_rng": action_space.np_random.bit_generator.state,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               temporary)
    temporary.replace(path)


def train(args, seed, device):
    set_seed(seed)
    env = gym.make(args.env)
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("Expected a Box observation space")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("TD3 requires a continuous Box action space")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        raise ValueError("This example expects one-dimensional observation/action arrays")
    env.action_space.seed(seed)
    agent = TD3(obs_dim, env.action_space.low, env.action_space.high,
                device, args, seed + 101)
    replay = ReplayBuffer(args.replay_capacity, obs_dim, action_dim, seed + 202)
    folder = args.out / args.env / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = [], []
    obs, _ = env.reset(seed=seed)
    episode_return, episode_length, episode = 0.0, 0, 0
    next_evaluation = args.eval_every
    next_checkpoint = args.checkpoint_every
    start_step = 0
    best_return = -float("inf")
    if args.resume:
        if len(args.seeds) != 1:
            raise ValueError("--resume requires exactly one seed")
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("environment") != args.env or int(saved.get("seed", -1)) != seed:
            raise ValueError("Resume checkpoint environment/seed does not match this run")
        if int(saved["step"]) >= args.steps:
            raise ValueError("Checkpoint already reached the requested --steps")
        agent.load_training_state(saved["agent"])
        replay.load_state_dict(saved["replay"])
        start_step = int(saved["step"])
        episode = int(saved["episode"])
        obs = np.asarray(saved["obs"], dtype=np.float32)
        train_rows, eval_rows = saved["train_rows"], saved["eval_rows"]
        best_return = float(saved["best_return"])
        next_evaluation = int(saved["next_evaluation"])
        next_checkpoint = int(saved["next_checkpoint"])
        env.action_space.np_random.bit_generator.state = saved["action_rng"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"Resuming {args.env} seed={seed} from step={start_step}", flush=True)
    try:
        for step in range(start_step + 1, args.steps + 1):
            if step <= args.warmup_steps:
                action = env.action_space.sample()
                normalized = np.clip((action - agent.mid) / agent.scale,
                                     -1, 1).astype(np.float32)
            else:
                normalized = agent.select_normalized(obs, explore=True)
                action = agent.to_environment_action(normalized)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(obs, normalized, reward, next_obs, terminated)
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1
            if step > args.warmup_steps and replay.size >= args.batch_size:
                agent.update(replay.sample(args.batch_size, device))
            episode_ended = bool(terminated or truncated)
            if episode_ended:
                episode += 1
                train_rows.append({"episode": episode, "env_steps": step,
                                   "return": episode_return, "length": episode_length,
                                   "terminated": bool(terminated),
                                   "truncated": bool(truncated)})
                obs, _ = env.reset(seed=seed + episode)
                episode_return, episode_length = 0.0, 0
            should_evaluate = step >= next_evaluation and (episode_ended or step == args.steps)
            if should_evaluate:
                result = aggregate_evaluation(
                    evaluate(agent, args.env, 100_000 + seed * 1000,
                             args.eval_episodes), step)
                eval_rows.append(result)
                print(f"{args.env} seed={seed} step={step} "
                      f"return={result['mean_return']:.2f} "
                      f"length={result['mean_length']:.1f} "
                      f"success={result['success_rate']:.2f}", flush=True)
                if result["mean_return"] > best_return:
                    best_return = result["mean_return"]
                    agent.save_policy(folder / "best.pt", args.env, seed, step)
                write_csv(folder / "train.csv", train_rows, TRAIN_FIELDS)
                write_csv(folder / "eval.csv", eval_rows, EVAL_FIELDS)
                next_evaluation += args.eval_every
            if episode_ended and step >= next_checkpoint:
                save_resume_checkpoint(
                    folder / "resume.pt", env_name=args.env, seed=seed,
                    step=step, episode=episode, obs=obs, agent=agent,
                    replay=replay, train_rows=train_rows, eval_rows=eval_rows,
                    best_return=best_return, next_evaluation=next_evaluation,
                    next_checkpoint=step + args.checkpoint_every,
                    action_space=env.action_space)
                next_checkpoint = step + args.checkpoint_every
    finally:
        env.close()

    agent.save_policy(folder / "last.pt", args.env, seed, args.steps)
    best_policy = torch.load(folder / "best.pt", map_location=device, weights_only=True)
    agent.actor.load_state_dict(best_policy["actor"])
    holdout_rows = evaluate(agent, args.env, args.holdout_seed + seed * 1000,
                            args.holdout_episodes)
    holdout = aggregate_evaluation(holdout_rows, int(best_policy["env_steps"]))
    holdout.update({"test_seed": args.holdout_seed + seed * 1000,
                    "test_episodes": args.holdout_episodes,
                    "checkpoint": "best.pt",
                    "selected_at_env_steps": int(best_policy["env_steps"])})
    (folder / "holdout.json").write_text(
        json.dumps(holdout, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(folder / "holdout_episodes.csv", holdout_rows,
              ("return", "length", "success", "upright_fraction"))
    baseline_summaries = []
    if args.env == "Pendulum-v1":
        baseline_rows = []
        for controller in ("random", "pd"):
            result, rows = evaluate_baseline(
                args.env, controller, args.holdout_seed + seed * 1000,
                args.holdout_episodes)
            baseline_summaries.append(result)
            baseline_rows.extend(dict(controller=controller, **row) for row in rows)
        (folder / "baselines.json").write_text(
            json.dumps(baseline_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(folder / "baseline_episodes.csv", baseline_rows,
                  ("controller", "return", "length", "success", "upright_fraction"))
    print(f"{args.env} seed={seed} holdout={holdout['mean_return']:.2f} "
          f"± {holdout['std_return']:.2f} success={holdout['success_rate']:.2f}", flush=True)
    summary = {"algorithm": "TD3", "environment": args.env, "seed": seed,
               "env_steps": args.steps, "episodes_completed": episode,
               "updates": agent.updates, "device": str(device),
               "torch_version": torch.__version__,
               "gymnasium_version": gym.__version__,
               "numpy_version": np.__version__,
               "evaluation_episodes": args.eval_episodes,
               "final_evaluation": eval_rows[-1],
               "holdout": holdout,
               "baselines": baseline_summaries,
               "best_mean_return": best_return,
               "hyperparameters": {key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()
                                   if key not in ("out", "seeds", "resume")}}
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS,
                        default="InvertedPendulum-v5")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--holdout-episodes", type=int, default=100)
    parser.add_argument("--holdout-seed", type=int, default=300_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume from a resume.pt snapshot saved at an episode boundary")
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results_td3"))
    args = parser.parse_args()
    positive = (args.steps, args.eval_every, args.eval_episodes,
                args.holdout_episodes, args.checkpoint_every,
                args.batch_size, args.replay_capacity, args.policy_delay,
                args.torch_threads)
    if any(value <= 0 for value in positive) or args.warmup_steps < 0:
        parser.error("counts must be positive; warmup-steps may be zero")
    if args.batch_size > args.replay_capacity:
        parser.error("batch-size must not exceed replay-capacity")
    if not (0 < args.gamma <= 1 and 0 < args.tau <= 1):
        parser.error("gamma and tau must be in (0, 1]")
    if any(value < 0 for value in (args.exploration_noise,
                                   args.policy_noise, args.noise_clip)):
        parser.error("noise parameters must be nonnegative")
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = [train(args, seed, device) for seed in args.seeds]
    if len(summaries) > 1:
        scores = np.asarray([item["final_evaluation"]["mean_return"]
                             for item in summaries])
        heldout = np.asarray([item["holdout"]["mean_return"] for item in summaries])
        aggregate = {"environment": args.env, "seeds": args.seeds,
                     "final_mean_return_across_seeds": float(scores.mean()),
                     "final_std_across_seeds": float(scores.std(ddof=1)),
                     "holdout_mean_return_across_seeds": float(heldout.mean()),
                     "holdout_std_across_seeds": float(heldout.std(ddof=1)),
                     "holdout_mean_ci95_t": mean_ci95(heldout),
                     "n_training_seeds": len(heldout)}
        if args.env == "Pendulum-v1":
            aggregate["baselines_across_seeds"] = {}
            for controller in ("random", "pd"):
                values = np.asarray([
                    next(item for item in summary["baselines"]
                         if item["controller"] == controller)["mean_return"]
                    for summary in summaries], dtype=np.float64)
                aggregate["baselines_across_seeds"][controller] = {
                    "mean_return": float(values.mean()),
                    "std_across_seeds": float(values.std(ddof=1)),
                    "mean_ci95_t": mean_ci95(values)}
        destination = args.out / args.env / "aggregate.json"
        destination.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"Across seeds: {aggregate}")


if __name__ == "__main__":
    main()
