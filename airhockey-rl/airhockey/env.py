"""Gymnasium env. The agent controls the BOTTOM paddle; the TOP paddle
is driven by `self.opponent`, a callable obs_top -> action in [-1, 1].

Observation: (10,) float32 — see physics.get_obs.
Action: (2,) float32 in [-1, 1], scaled to physics.max_paddle_accel.

Reward (defaults):
    +10.0 on scoring, -10.0 on conceding, +0.1 on bot paddle contact,
    + shaping_coef * (-puck_vy / max_puck_speed) per step
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
        reward_hit: float = 0.3,
        reward_vel_coef: float = 0.02,
        reward_puck_side_coef: float = 0.005,
        reward_approach_coef: float = 0.5,
        reward_time_coef: float = 0.001,
        max_episode_steps: int = 800,
        seed: int = 0,
    ):
        super().__init__()
        self.physics = AirHockeyPhysics(physics_config, seed=seed)
        self.opponent = opponent
        self.reward_score = reward_score
        self.reward_concede = reward_concede
        self.reward_hit = reward_hit
        self.reward_vel_coef = reward_vel_coef
        self.reward_puck_side_coef = reward_puck_side_coef
        self.reward_approach_coef = reward_approach_coef
        self.reward_time_coef = reward_time_coef
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._prev_dist = None  # bot-paddle <-> puck distance at previous step

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return a * self.physics.cfg.max_paddle_accel

    def _opponent_action(self) -> np.ndarray:
        """The opponent sees a vertically mirrored obs so a bot-trained
        policy can drive the top paddle. Its output is in that mirrored
        frame, so we flip the y component back to world coordinates."""
        if self.opponent is None:
            return np.zeros(2, dtype=np.float32)
        obs_top = self.physics.get_obs(perspective="top")
        a = np.asarray(self.opponent(obs_top), dtype=np.float32).copy()
        a[1] = -a[1]
        return a * self.physics.cfg.max_paddle_accel

    def _puck_distance(self) -> float:
        s = self.physics.state
        return float(np.hypot(s.puck_x - s.bot_x, s.puck_y - s.bot_y))

    def _shaping_reward(self) -> float:
        s = self.physics.state
        c = self.physics.cfg
        # Puck velocity toward opponent goal (up = negative vy).
        v_norm = -s.puck_vy / c.max_puck_speed
        # Puck position bias: +1 at opponent wall, -1 at our wall.
        side = 1.0 - 2.0 * (s.puck_y / c.height)
        # Approach: reward the bot paddle for reducing its distance to
        # the puck. This gives a gradient even when the puck is at rest.
        cur_dist = self._puck_distance()
        if self._prev_dist is None:
            approach = 0.0
        else:
            delta = self._prev_dist - cur_dist  # positive when approaching
            approach = delta / c.max_paddle_speed
        self._prev_dist = cur_dist
        return (
            self.reward_vel_coef * v_norm
            + self.reward_puck_side_coef * side
            + self.reward_approach_coef * approach
            - self.reward_time_coef
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.physics.rng = np.random.default_rng(seed)
        # Randomize which side serves so the agent can't condition on
        # "always get first strike".
        serve = "bot" if self.physics.rng.random() < 0.5 else "top"
        self.physics.hard_reset(serve_to=serve)
        self._step_count = 0
        self._prev_dist = None
        obs = self.physics.get_obs(perspective="bot")
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        bot_accel = self._denorm_action(action)
        top_accel = self._opponent_action()
        event = self.physics.step(top_accel=top_accel, bot_accel=bot_accel)

        # physics.py convention: "goal_X" means paddle X scored. The agent
        # is the bot paddle, so goal_bot is a win and goal_top is a loss.
        reward = 0.0
        terminated = False
        if event == "goal_bot":
            reward += self.reward_score
            terminated = True
        elif event == "goal_top":
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
