"""Gymnasium env. The agent controls the BOTTOM paddle; the TOP paddle
is driven by `self.opponent`, a callable obs_top -> action in [-1, 1].

Observation: (10,) float32 — see physics.get_obs.
Action: (2,) float32 in [-1, 1], scaled to physics.max_paddle_accel.

Reward — dense micro-rewards every step that tell the agent what good
air hockey looks like, plus sparse +10/-10 on goals:

  +10 on scoring, -10 on conceding (sparse, dominant).

  Per step:
    - Closer to puck → small positive
    - Farther from puck → small negative
    - Hit the puck → small positive
    - Puck moving toward opponent goal → small positive
    - Puck moving toward own goal → small negative
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from airhockey.physics import AirHockeyPhysics, PhysicsConfig


OpponentFn = Callable[[np.ndarray], np.ndarray]


class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(
        self,
        physics_config: Optional[PhysicsConfig] = None,
        opponent: Optional[OpponentFn] = None,
        reward_score: float = 10.0,
        reward_concede: float = -10.0,
        max_episode_steps: int = 800,
        seed: int = 0,
    ):
        super().__init__()
        self.physics = AirHockeyPhysics(physics_config, seed=seed)
        self.opponent = opponent
        self.reward_score = reward_score
        self.reward_concede = reward_concede
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._prev_dist = 0.0
        self._last_scorer = ""

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _opponent_action(self) -> np.ndarray:
        if self.opponent is None:
            return np.zeros(2, dtype=np.float32)
        obs_top = self.physics.get_obs(perspective="top")
        a = np.asarray(self.opponent(obs_top), dtype=np.float32).copy()
        a[1] = -a[1]
        return a * self.physics.cfg.max_paddle_accel

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.physics.rng = np.random.default_rng(seed)
        if self._last_scorer == "":
            serve = "bot" if self.physics.rng.random() < 0.5 else "top"
        elif self._last_scorer == "goal_bot":
            serve = "top"
        else:
            serve = "bot"
        self._last_scorer = ""
        self.physics.hard_reset(serve_to=serve)
        self._step_count = 0
        s = self.physics.state
        self._prev_dist = float(np.hypot(s.puck_x - s.bot_x, s.puck_y - s.bot_y))
        obs = self.physics.get_obs(perspective="bot")
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        action_arr = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        bot_accel = action_arr * self.physics.cfg.max_paddle_accel
        top_accel = self._opponent_action()

        s0 = self.physics.state
        pre_puck = (s0.puck_x, s0.puck_y, s0.puck_vx, s0.puck_vy)

        event = self.physics.step(top_accel=top_accel, bot_accel=bot_accel)

        reward = 0.0
        terminated = False

        # Sparse goal reward
        if event == "goal_bot":
            reward = self.reward_score
            terminated = True
            self._last_scorer = "goal_bot"
        elif event == "goal_top":
            reward = self.reward_concede
            terminated = True
            self._last_scorer = "goal_top"

        # Dense micro-rewards (always on)
        if not terminated:
            s = self.physics.state
            c = self.physics.cfg

            # 1. Approach reward: getting closer to puck is good
            cur_dist = float(np.hypot(s.puck_x - s.bot_x, s.puck_y - s.bot_y))
            approach = (self._prev_dist - cur_dist) / c.max_paddle_speed
            reward += 0.1 * approach
            self._prev_dist = cur_dist

            # 2. Puck direction: puck moving toward opponent goal is good
            #    (negative vy = toward top = good for bot)
            reward += 0.02 * (-s.puck_vy / c.max_puck_speed)

        # Hit bonus
        if event == "hit_bot":
            reward += 0.2

        truncated = self._step_count >= self.max_episode_steps
        obs = self.physics.get_obs(perspective="bot")
        info = {
            "event": event,
            "top_score": self.physics.state.top_score,
            "bot_score": self.physics.state.bot_score,
        }
        if event.startswith("goal"):
            info["goal_puck"] = pre_puck
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None
