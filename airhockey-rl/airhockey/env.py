"""Gymnasium env. The agent controls the BOTTOM paddle; the TOP paddle
is driven by `self.opponent`, a callable obs_top -> action in [-1, 1].

Observation: (10,) float32 — see physics.get_obs.
Action: (2,) float32 in [-1, 1], scaled to physics.max_paddle_accel.

Reward — pure sparse by default (+10 on scoring, -10 on conceding,
0 otherwise). Optionally a curriculum strike-shaping bonus on bot/puck
contact, gated by `shaping_enabled` so the trainer can enable it
during a curriculum's early phase and disable it after.

Curriculum shaping has two parts, both designed around how a human
plays air hockey:

  STRIKE QUALITY (with bank-shot awareness): when the bot hits the
  puck on the opponent's half, forward-project the puck's outgoing
  trajectory to the opponent paddle's y-line. If the linear projection
  would hit a side wall first, reflect once across that wall. The
  shot quality is the predicted clearance from the opponent paddle
  at that line. This rewards angled shots AND bank shots, not just
  straight-on shots.

  LATERAL APPROACH: also on a hit_bot event, reward the magnitude of
  the bot paddle's *lateral* (x-direction) velocity at the moment of
  contact. A player approaches the puck sideways so the contact normal
  points where they want the puck to go; the agent has no other
  gradient telling it that "approach from the side" is the trick.

Random initial position for the bot paddle is always on (not tied to
shaping), because being off-center is normal in real play and the
agent should not assume it always starts on the central line.
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
        reward_lateral_coef: float = 0.3,
        max_episode_steps: int = 800,
        seed: int = 0,
    ):
        super().__init__()
        self.physics = AirHockeyPhysics(physics_config, seed=seed)
        self.opponent = opponent
        self.reward_score = reward_score
        self.reward_concede = reward_concede
        self.reward_strike_coef = reward_strike_coef
        self.reward_lateral_coef = reward_lateral_coef
        self.shaping_enabled = False  # train_sac.py flips this on/off
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _bank_aware_quality(
        self,
        puck_x: float, puck_y: float,
        puck_vx: float, puck_vy: float,
        top_x: float, top_y: float,
    ) -> float:
        """Forward-project the puck to the opponent paddle's y line,
        following at most one reflection off a side wall. Return the
        clearance from the opponent paddle, normalized to [0, 1] over
        2 paddle radii.

        This rewards both straight angled shots AND single-bounce bank
        shots, which the original linear-only quality metric did not.
        """
        c = self.physics.cfg
        x1 = self.physics.field_x1
        x2 = self.physics.field_x2

        # We need puck to actually be moving toward the opponent.
        if puck_vy >= -1e-6:
            return 0.0

        # First, check if a straight-line projection would hit a side
        # wall before reaching y = top_y. The straight-line travel time
        # to the opponent paddle line:
        t_total = (puck_y - top_y) / (-puck_vy)
        if t_total <= 0:
            return 0.0

        # Time to hit a side wall (if the puck is moving toward one).
        t_wall = float("inf")
        wall_x = None
        if puck_vx > 1e-6:
            t_w = (x2 - puck_x) / puck_vx
            if 0 < t_w < t_wall:
                t_wall = t_w
                wall_x = x2
        elif puck_vx < -1e-6:
            t_w = (x1 - puck_x) / puck_vx
            if 0 < t_w < t_wall:
                t_wall = t_w
                wall_x = x1

        if t_wall >= t_total:
            # Direct shot — no bounce before reaching the opponent.
            pred_x = puck_x + puck_vx * t_total
        else:
            # Reflect once off the side wall, continue to opponent line.
            x_at_wall = wall_x  # by construction
            y_at_wall = puck_y + puck_vy * t_wall
            # After reflection, x velocity flips (with restitution).
            new_vx = -puck_vx * c.e_wall
            t_remain = (y_at_wall - top_y) / (-puck_vy)
            if t_remain <= 0:
                return 0.0
            pred_x = x_at_wall + new_vx * t_remain
            # If the bounced trajectory would *also* leave the field,
            # clip rather than recurse — one bounce is enough for the
            # signal we want.
            pred_x = float(np.clip(pred_x, x1, x2))

        clearance = abs(pred_x - top_x)
        return float(min(1.0, clearance / (2.0 * c.paddle_radius)))

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

        # Random non-home start positions for both paddles. Real players
        # are rarely on the central line at the start of a rally; if the
        # agent always starts dead-center, the shortest path to the puck
        # is always head-on, which produces straight shots into the
        # opponent's center. Off-center starts force the agent to learn
        # that being off-axis is normal.
        s = self.physics.state
        rng = self.physics.rng
        s.bot_x = float(rng.uniform(self.physics.paddle_min_x, self.physics.paddle_max_x))
        s.bot_y = float(rng.uniform(self.physics.bot_min_y, self.physics.bot_max_y))
        s.top_x = float(rng.uniform(self.physics.paddle_min_x, self.physics.paddle_max_x))
        s.top_y = float(rng.uniform(self.physics.top_min_y, self.physics.top_max_y))

        self._step_count = 0
        obs = self.physics.get_obs(perspective="bot")
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        action_arr = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        bot_accel = action_arr * self.physics.cfg.max_paddle_accel
        top_accel = self._opponent_action()

        # Snapshot puck pre-step so we can log the goal-shot location.
        # Snapshot bot velocity pre-step so the lateral-approach reward
        # uses the actual approach velocity (post-collision the elastic
        # bounce has already modified bot_vx by the impulse).
        s0 = self.physics.state
        pre_puck = (s0.puck_x, s0.puck_y, s0.puck_vx, s0.puck_vy)
        pre_bot_vx = s0.bot_vx

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

        # Curriculum strike shaping. Phase 1 only. Two parts:
        #
        #   1. Strike quality (with one-bounce bank-shot awareness):
        #      reward outgoing puck speed times the predicted clearance
        #      from the opponent paddle. The forward-projection follows
        #      one wall reflection if needed, so bank shots count.
        #
        #   2. Lateral-approach bonus: reward the magnitude of the bot
        #      paddle's pre-collision lateral velocity. This is the
        #      gradient that teaches "approach the puck from the side
        #      so the contact normal points where you want it to go,"
        #      which is the geometric trick a real player uses.
        #
        # Both gates are also conditional on the contact happening on
        # the opponent's half — defensive clears do not earn shaping.
        if self.shaping_enabled and event == "hit_bot":
            ps = self.physics.state
            pc = self.physics.cfg
            on_opponent_half = ps.puck_y < pc.height / 2
            if on_opponent_half:
                outgoing = max(0.0, -ps.puck_vy / pc.max_puck_speed)
                quality = 0.0
                if (
                    self.reward_strike_coef > 0.0
                    and outgoing > 0.0
                    and ps.puck_vy < -1e-3
                ):
                    quality = self._bank_aware_quality(
                        ps.puck_x, ps.puck_y, ps.puck_vx, ps.puck_vy,
                        ps.top_x, ps.top_y,
                    )
                reward += self.reward_strike_coef * outgoing * quality

                if self.reward_lateral_coef > 0.0:
                    lateral = abs(pre_bot_vx) / pc.max_paddle_speed
                    reward += self.reward_lateral_coef * lateral

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
