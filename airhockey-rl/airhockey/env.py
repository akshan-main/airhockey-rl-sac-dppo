"""Gymnasium env. The agent controls the BOTTOM paddle; the TOP paddle
is driven by `self.opponent`, a callable obs_top -> action in [-1, 1].

Observation: (10,) float32 — see physics.get_obs.
Action: (2,) float32 in [-1, 1], scaled to physics.max_paddle_accel.

Reward — pure sparse by default (+10 on scoring, -10 on conceding,
0 otherwise). Optionally a small strike-shaping bonus on bot/puck
contact, gated by `shaping_enabled` so the trainer can enable it
during a curriculum's early phase and disable it after. The bonus is
proportional to outgoing puck speed * shot quality (forward-projected
clearance from the opponent paddle), so it rewards aimed shots, not
random taps.
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
        reward_strike_coef: float = 0.5,
        max_episode_steps: int = 800,
        seed: int = 0,
    ):
        super().__init__()
        self.physics = AirHockeyPhysics(physics_config, seed=seed)
        self.opponent = opponent
        self.reward_score = reward_score
        self.reward_concede = reward_concede
        self.reward_strike_coef = reward_strike_coef
        self.shaping_enabled = False  # train_sac.py flips this on/off
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

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
        obs = self.physics.get_obs(perspective="bot")
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        action_arr = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        bot_accel = action_arr * self.physics.cfg.max_paddle_accel
        top_accel = self._opponent_action()

        # Snapshot puck pre-step so we can log the goal-shot location.
        s0 = self.physics.state
        pre_puck = (s0.puck_x, s0.puck_y, s0.puck_vx, s0.puck_vy)

        event = self.physics.step(top_accel=top_accel, bot_accel=bot_accel)

        # Sparse reward (always on). physics.py convention: "goal_X"
        # means paddle X scored. The bot is the agent so goal_bot = win.
        reward = 0.0
        terminated = False
        if event == "goal_bot":
            reward = self.reward_score
            terminated = True
        elif event == "goal_top":
            reward = self.reward_concede
            terminated = True

        # Curriculum strike shaping. Only fires if the trainer enabled
        # it (Phase 1 of the curriculum), the bot just hit the puck,
        # AND the contact happened on the opponent's half — i.e. the
        # agent had to cross midfield to attack. This rules out
        # collecting strike bonus on defensive clears.
        if (
            self.shaping_enabled
            and event == "hit_bot"
            and self.reward_strike_coef > 0.0
        ):
            ps = self.physics.state
            pc = self.physics.cfg
            on_opponent_half = ps.puck_y < pc.height / 2
            if on_opponent_half:
                outgoing = max(0.0, -ps.puck_vy / pc.max_puck_speed)
                quality = 0.0
                if outgoing > 0.0 and ps.puck_vy < -1e-3:
                    tof = max(0.0, (ps.puck_y - ps.top_y) / (-ps.puck_vy))
                    pred_x = ps.puck_x + ps.puck_vx * tof
                    clearance = abs(pred_x - ps.top_x)
                    quality = min(1.0, clearance / (2.0 * pc.paddle_radius))
                reward += self.reward_strike_coef * outgoing * quality

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
