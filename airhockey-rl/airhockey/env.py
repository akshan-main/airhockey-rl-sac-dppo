"""Gymnasium-compatible air hockey environment.

Single-agent: the policy controls the BOTTOM paddle. The TOP paddle is
driven by an opponent — any callable that takes a top-perspective 10-D
observation and returns a 2-D normalized action in [-1, 1]. In practice
the opponent is loaded via `airhockey.snapshot_opponent.load_opponent`
from either a trained SAC checkpoint or a trained Diffusion Policy
checkpoint, but the env is agnostic to the opponent's implementation.

Observation: 10D float32 in [-1, 1] / [0, 1].
Action: 2D continuous in [-1, 1] interpreted as (ax, ay) accel scaled to
        physics.max_paddle_accel.

Reward shaping (configurable):
    +10  on scoring (puck enters opponent goal)
    -10  on conceding
    +0.1 on each puck contact by our paddle
    +0.01 * v_puck_y_normalized   per step (encourages pushing puck up)
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from airhockey.physics import AirHockeyPhysics, PhysicsConfig


OpponentFn = Callable[[np.ndarray], np.ndarray]
"""Opponent: takes a 10D observation (top-paddle perspective) and returns
a 2D normalized action in [-1, 1]."""


class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(
        self,
        physics_config: Optional[PhysicsConfig] = None,
        opponent: Optional[OpponentFn] = None,
        reward_score: float = 10.0,
        reward_concede: float = -10.0,
        reward_hit: float = 0.1,
        reward_shaping_coef: float = 0.01,
        max_episode_steps: int = 1500,
        seed: int = 0,
    ):
        super().__init__()
        self.physics = AirHockeyPhysics(physics_config, seed=seed)
        self.opponent = opponent
        self.reward_score = reward_score
        self.reward_concede = reward_concede
        self.reward_hit = reward_hit
        self.reward_shaping_coef = reward_shaping_coef
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    # ── Helpers ───────────────────────────────────────────────
    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return a * self.physics.cfg.max_paddle_accel

    def _opponent_action(self) -> np.ndarray:
        """Get the opponent's action, denormalized to physics accel units.

        The opponent callable receives the top paddle's perspective obs
        (vertically mirrored, so the policy trained as the bottom paddle
        "sees itself" in its native frame). The policy's output action is
        interpreted in that same mirrored frame, so to apply it to the
        world-frame top paddle we flip the y component back.
        """
        if self.opponent is None:
            return np.zeros(2, dtype=np.float32)
        obs_top = self.physics.get_obs(perspective="top")
        a = np.asarray(self.opponent(obs_top), dtype=np.float32).copy()
        a[1] = -a[1]  # mirror y back into world frame
        return a * self.physics.cfg.max_paddle_accel

    def _shaping_reward(self) -> float:
        s = self.physics.state
        # Encourage pushing puck toward the top (opponent) goal — i.e. negative y velocity
        v_norm = -s.puck_vy / self.physics.cfg.max_puck_speed
        return self.reward_shaping_coef * v_norm

    # ── Gym API ───────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.physics.rng = np.random.default_rng(seed)
        self.physics.hard_reset(serve_to="bot")
        self._step_count = 0
        obs = self.physics.get_obs(perspective="bot")
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        bot_accel = self._denorm_action(action)
        top_accel = self._opponent_action()
        event = self.physics.step(top_accel=top_accel, bot_accel=bot_accel)

        reward = 0.0
        terminated = False
        if event == "goal_top":
            reward += self.reward_score
            terminated = True
        elif event == "goal_bot":
            reward += self.reward_concede
            terminated = True
        elif event == "hit_bot":
            reward += self.reward_hit

        reward += self._shaping_reward()

        truncated = self._step_count >= self.max_episode_steps
        obs = self.physics.get_obs(perspective="bot")
        info = {"event": event, "top_score": self.physics.state.top_score,
                "bot_score": self.physics.state.bot_score}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None
