"""Gymnasium env. The agent controls the BOTTOM paddle; the TOP paddle
is driven by `self.opponent`, a callable obs_top -> action in [-1, 1].

Observation: (10,) float32 — see physics.get_obs.
Action: (2,) float32 in [-1, 1], scaled to physics.max_paddle_accel.

Reward shape — designed to teach defend → strike → return:

  +10.0 on scoring, -10.0 on conceding (sparse, dominant).

  STRIKE event: on a bot/puck contact, bonus proportional to the puck's
  outgoing speed *toward the opponent goal*. A hard shot is worth a lot
  more than a tap, so the agent has an incentive to wind up rather than
  just bump.

  Per-step shaping is conditional on puck side:
    - Puck on the bot's defensive half → small reward for closing the
      distance to the puck (defend / intercept).
    - Puck on the opponent's half → small reward for being near home y
      (retreat / reset position after a strike).

  Plus a tiny time penalty so dawdling isn't free. All shaping terms
  are small enough that they accumulate to roughly ±2 over a full
  800-step episode, so the sparse ±10 goal reward stays in charge.
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
        reward_approach_coef: float = 0.05,
        reward_home_coef: float = 0.002,
        reward_time_coef: float = 0.001,
        reward_jerk_coef: float = 0.02,
        max_episode_steps: int = 800,
        seed: int = 0,
    ):
        super().__init__()
        self.physics = AirHockeyPhysics(physics_config, seed=seed)
        self.opponent = opponent
        self.reward_score = reward_score
        self.reward_concede = reward_concede
        self.reward_strike_coef = reward_strike_coef
        self.reward_approach_coef = reward_approach_coef
        self.reward_home_coef = reward_home_coef
        self.reward_time_coef = reward_time_coef
        self.reward_jerk_coef = reward_jerk_coef
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._prev_dist = None  # bot-paddle <-> puck distance at previous step
        self._prev_action = None  # last commanded action (for jerk penalty)

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

    def _puck_distance(self) -> float:
        s = self.physics.state
        return float(np.hypot(s.puck_x - s.bot_x, s.puck_y - s.bot_y))

    def _shaping_reward(self) -> float:
        """Per-step shaping. Conditional on which half the puck is in.

        Defensive half (puck_y > height/2): reward closing distance to
        the puck, so the agent learns to intercept.
        Offensive half (puck_y < height/2): reward being near home y, so
        the agent learns to retreat after a strike instead of camping
        forward.
        """
        s = self.physics.state
        c = self.physics.cfg
        on_bot_half = s.puck_y > c.height / 2

        cur_dist = self._puck_distance()
        if self._prev_dist is None or not on_bot_half:
            approach = 0.0
        else:
            delta = self._prev_dist - cur_dist  # positive when approaching
            approach = delta / c.max_paddle_speed
        self._prev_dist = cur_dist

        if on_bot_half:
            home_term = 0.0
        else:
            # Distance from home y, normalized to [0, 1]. We use the
            # paddle's max-y position (its closest-to-wall point) as
            # "home" — a defender's resting spot.
            home_err = abs(s.bot_y - self.physics.bot_max_y) / c.height
            home_term = -home_err

        return (
            self.reward_approach_coef * approach
            + self.reward_home_coef * home_term
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
        self._prev_action = None
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
            # Strike bonus: outgoing puck speed * shot quality. The
            # quality term measures how well the shot clears the
            # opponent paddle, so a hard shot fired straight at the
            # opponent gets ~0 reward and a hard shot angled around
            # them gets up to ~1. Without this multiplier the agent
            # has no aiming gradient and learns to fire at the center
            # — which is exactly where a stationary opponent sits.
            ps = self.physics.state
            pc = self.physics.cfg
            outgoing = max(0.0, -ps.puck_vy / pc.max_puck_speed)
            quality = 0.0
            if outgoing > 0.0 and ps.puck_vy < -1e-3:
                # Linear forward-extrapolation to the opponent's y. Time
                # of flight to the top paddle's current y line.
                tof = max(0.0, (ps.puck_y - ps.top_y) / (-ps.puck_vy))
                pred_x = ps.puck_x + ps.puck_vx * tof
                clearance = abs(pred_x - ps.top_x)
                # Normalize: 0 at the paddle's center, 1.0 once the
                # predicted miss is >= 2 paddle radii.
                quality = min(1.0, clearance / (2.0 * pc.paddle_radius))
            reward += self.reward_strike_coef * outgoing * quality

        reward += self._shaping_reward()

        # Jerk penalty: -coef * ||a_t - a_{t-1}||^2. Encourages smooth,
        # human-looking paddle motion instead of frame-to-frame thrash.
        if self._prev_action is not None:
            jerk = float(np.sum((action_arr - self._prev_action) ** 2))
            reward -= self.reward_jerk_coef * jerk
        self._prev_action = action_arr

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
