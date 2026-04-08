"""AirHockey2D-v0 — third-party env contribution to Farama Foundation.

This is a polished, dependency-minimal copy of `airhockey/env.py` formatted
to match the Farama style guide for third-party Gymnasium environments.
The training code in the main repo imports from `airhockey.env` directly;
this file is the standalone version proposed for the registry.

Differences from the in-tree version:
  • Type hints on every public method
  • Docstrings in NumPy format
  • Imports only from numpy + gymnasium (no torch, no fastapi)
  • The env class has zero opinion about the opponent — any callable
    that maps a 10-D observation to a 2-D normalized action will work
  • Standard `metadata` dict with render modes
  • Pytest-friendly determinism via `gymnasium.utils.seeding`
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


# ── Physics constants ──────────────────────────────────────────
WIDTH = 320.0
HEIGHT = 540.0
WALL = 12.0
GOAL_WIDTH = 110.0
PUCK_RADIUS = 11.0
PADDLE_RADIUS = 19.0
PUCK_MASS = 1.0
PADDLE_MASS = 4.0
E_WALL = 0.94
E_PADDLE = 0.96
DRAG = 0.18
MAX_PUCK_SPEED = 950.0
MAX_PADDLE_SPEED = 700.0
MAX_PADDLE_ACCEL = 4500.0
DT = 1.0 / 50.0
SUBSTEPS = 3


OpponentFn = Callable[[np.ndarray], np.ndarray]


class AirHockey2DEnv(gym.Env):
    """Custom 2D top-down air hockey environment.

    A bottom paddle (controlled by the agent) plays against a top paddle
    (controlled by an opponent function). The puck moves under linear
    drag with elastic wall reflection. Paddle/puck contact uses
    line-of-impact elastic collision with mass ratio 4:1.

    Parameters
    ----------
    opponent : callable, optional
        Function ``opponent(obs_top) -> action`` returning a normalized
        2D action in [-1, 1]. ``obs_top`` is the 10-D observation in
        the top paddle's perspective. Defaults to a no-op opponent.
    max_episode_steps : int, default 1500
        Hard episode timeout.
    reward_score : float, default 10.0
        Reward for scoring on the opponent.
    reward_concede : float, default -10.0
        Reward for being scored on.
    reward_hit : float, default 0.1
        Reward for the agent's paddle making contact with the puck.
    reward_shaping_coef : float, default 0.01
        Coefficient on the per-step velocity-toward-opponent shaping reward.

    Attributes
    ----------
    observation_space : spaces.Box
        10-D Box in [-1, 1] (positions normalized by rink dims, velocities
        by max speed).
    action_space : spaces.Box
        2-D Box in [-1, 1] interpreted as (ax, ay) accel, scaled internally
        to ``MAX_PADDLE_ACCEL``.

    Notes
    -----
    The env is fully observed and deterministic given seed + opponent.
    All physics is integrated with semi-implicit (symplectic) Euler at a
    sub-step of ``DT / SUBSTEPS`` for stability.
    """

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(
        self,
        opponent: Optional[OpponentFn] = None,
        max_episode_steps: int = 1500,
        reward_score: float = 10.0,
        reward_concede: float = -10.0,
        reward_hit: float = 0.1,
        reward_shaping_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.opponent = opponent
        self.max_episode_steps = max_episode_steps
        self.reward_score = reward_score
        self.reward_concede = reward_concede
        self.reward_hit = reward_hit
        self.reward_shaping_coef = reward_shaping_coef

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._step_count = 0
        self._np_random, _ = seeding.np_random(0)
        self._init_state()

    # ── Lifecycle ─────────────────────────────────────────────
    def _init_state(self) -> None:
        self.puck = np.array([WIDTH / 2, HEIGHT * 0.72, 0.0, 0.0], dtype=np.float64)
        self.top = np.array([WIDTH / 2, WALL + PADDLE_RADIUS + 60, 0.0, 0.0], dtype=np.float64)
        self.bot = np.array([WIDTH / 2, HEIGHT - WALL - PADDLE_RADIUS - 60, 0.0, 0.0], dtype=np.float64)
        self.top_score = 0
        self.bot_score = 0
        self._last_event = ""

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)
        self._init_state()
        self._step_count = 0
        return self._obs("bot"), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) * MAX_PADDLE_ACCEL
        opp_action = self._opponent_action()

        self._last_event = ""
        sub_dt = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            self._integrate_paddle(self.top, opp_action, sub_dt, is_top=True)
            self._integrate_paddle(self.bot, action, sub_dt, is_top=False)
            self._integrate_puck(sub_dt)
            self._collide(self.top, side="top")
            self._collide(self.bot, side="bot")
            if self._last_event.startswith("goal"):
                self._reset_after_goal()
                break

        reward = 0.0
        terminated = False
        if self._last_event == "goal_top":
            reward += self.reward_score
            terminated = True
        elif self._last_event == "goal_bot":
            reward += self.reward_concede
            terminated = True
        elif self._last_event == "hit_bot":
            reward += self.reward_hit

        # Shaping: encourage pushing the puck up (negative y velocity)
        reward += self.reward_shaping_coef * (-self.puck[3] / MAX_PUCK_SPEED)

        truncated = self._step_count >= self.max_episode_steps
        return self._obs("bot"), float(reward), terminated, truncated, {
            "event": self._last_event,
            "top_score": self.top_score,
            "bot_score": self.bot_score,
        }

    def render(self) -> None:
        return None

    # ── Helpers ───────────────────────────────────────────────
    def _opponent_action(self) -> np.ndarray:
        if self.opponent is None:
            return np.zeros(2, dtype=np.float32)
        a = np.asarray(self.opponent(self._obs("top")), dtype=np.float32).copy()
        a[1] = -a[1]  # mirror y back
        return a * MAX_PADDLE_ACCEL

    def _reset_after_goal(self) -> None:
        serve_to = "top" if self._last_event == "goal_bot" else "bot"
        self.puck[0] = WIDTH / 2
        self.puck[1] = HEIGHT * (0.28 if serve_to == "top" else 0.72)
        self.puck[2] = self.puck[3] = 0.0
        self.top[:] = [WIDTH / 2, WALL + PADDLE_RADIUS + 60, 0.0, 0.0]
        self.bot[:] = [WIDTH / 2, HEIGHT - WALL - PADDLE_RADIUS - 60, 0.0, 0.0]

    def _integrate_paddle(self, p: np.ndarray, accel: np.ndarray, dt: float, is_top: bool) -> None:
        ax, ay = float(accel[0]), float(accel[1])
        a_mag = float(np.hypot(ax, ay))
        if a_mag > MAX_PADDLE_ACCEL:
            ax = ax / a_mag * MAX_PADDLE_ACCEL
            ay = ay / a_mag * MAX_PADDLE_ACCEL
        p[2] += ax * dt
        p[3] += ay * dt
        sp = float(np.hypot(p[2], p[3]))
        if sp > MAX_PADDLE_SPEED:
            p[2] = p[2] / sp * MAX_PADDLE_SPEED
            p[3] = p[3] / sp * MAX_PADDLE_SPEED
        p[0] += p[2] * dt
        p[1] += p[3] * dt
        p[0] = float(np.clip(p[0], WALL + PADDLE_RADIUS, WIDTH - WALL - PADDLE_RADIUS))
        if is_top:
            ny = float(np.clip(p[1], WALL + PADDLE_RADIUS, HEIGHT / 2 - PADDLE_RADIUS))
        else:
            ny = float(np.clip(p[1], HEIGHT / 2 + PADDLE_RADIUS, HEIGHT - WALL - PADDLE_RADIUS))
        if ny != p[1]:
            p[3] = 0.0
            p[1] = ny

    def _integrate_puck(self, dt: float) -> None:
        damp = max(0.0, 1.0 - DRAG * dt)
        self.puck[2] *= damp
        self.puck[3] *= damp
        sp = float(np.hypot(self.puck[2], self.puck[3]))
        if sp > MAX_PUCK_SPEED:
            self.puck[2] = self.puck[2] / sp * MAX_PUCK_SPEED
            self.puck[3] = self.puck[3] / sp * MAX_PUCK_SPEED
        self.puck[0] += self.puck[2] * dt
        self.puck[1] += self.puck[3] * dt

        x1 = WALL + PUCK_RADIUS
        x2 = WIDTH - WALL - PUCK_RADIUS
        y1 = WALL + PUCK_RADIUS
        y2 = HEIGHT - WALL - PUCK_RADIUS
        gx1 = (WIDTH - GOAL_WIDTH) / 2
        gx2 = (WIDTH + GOAL_WIDTH) / 2

        if self.puck[0] < x1:
            self.puck[0] = x1
            self.puck[2] = -self.puck[2] * E_WALL
        elif self.puck[0] > x2:
            self.puck[0] = x2
            self.puck[2] = -self.puck[2] * E_WALL
        if self.puck[1] < y1:
            if gx1 + PUCK_RADIUS < self.puck[0] < gx2 - PUCK_RADIUS:
                self.bot_score += 1
                self._last_event = "goal_bot"
                return
            self.puck[1] = y1
            self.puck[3] = -self.puck[3] * E_WALL
        if self.puck[1] > y2:
            if gx1 + PUCK_RADIUS < self.puck[0] < gx2 - PUCK_RADIUS:
                self.top_score += 1
                self._last_event = "goal_top"
                return
            self.puck[1] = y2
            self.puck[3] = -self.puck[3] * E_WALL

    def _collide(self, p: np.ndarray, side: str) -> None:
        dx = self.puck[0] - p[0]
        dy = self.puck[1] - p[1]
        d = float(np.hypot(dx, dy))
        md = PUCK_RADIUS + PADDLE_RADIUS
        if d >= md or d < 1e-3:
            return
        nx = dx / d
        ny = dy / d
        v_rel_x = self.puck[2] - p[2]
        v_rel_y = self.puck[3] - p[3]
        v_rel_n = v_rel_x * nx + v_rel_y * ny
        if v_rel_n >= 0:
            return
        j = -(1 + E_PADDLE) * v_rel_n / (1 / PUCK_MASS + 1 / PADDLE_MASS)
        self.puck[2] += (j / PUCK_MASS) * nx
        self.puck[3] += (j / PUCK_MASS) * ny
        p[2] -= (j / PADDLE_MASS) * nx
        p[3] -= (j / PADDLE_MASS) * ny
        overlap = md - d
        self.puck[0] += nx * overlap
        self.puck[1] += ny * overlap
        self._last_event = f"hit_{side}"

    def _obs(self, perspective: str) -> np.ndarray:
        if perspective == "bot":
            return np.array([
                self.puck[0] / WIDTH,
                self.puck[1] / HEIGHT,
                self.puck[2] / MAX_PUCK_SPEED,
                self.puck[3] / MAX_PUCK_SPEED,
                self.bot[0] / WIDTH,
                self.bot[1] / HEIGHT,
                self.bot[2] / MAX_PADDLE_SPEED,
                self.bot[3] / MAX_PADDLE_SPEED,
                self.top[0] / WIDTH,
                self.top[1] / HEIGHT,
            ], dtype=np.float32)
        return np.array([
            self.puck[0] / WIDTH,
            1.0 - self.puck[1] / HEIGHT,
            self.puck[2] / MAX_PUCK_SPEED,
            -self.puck[3] / MAX_PUCK_SPEED,
            self.top[0] / WIDTH,
            1.0 - self.top[1] / HEIGHT,
            self.top[2] / MAX_PADDLE_SPEED,
            -self.top[3] / MAX_PADDLE_SPEED,
            self.bot[0] / WIDTH,
            1.0 - self.bot[1] / HEIGHT,
        ], dtype=np.float32)


def register() -> None:
    """Register `AirHockey2D-v0` with gymnasium so users can do
    ``gymnasium.make("AirHockey2D-v0")``."""
    from gymnasium.envs.registration import register as gym_register
    gym_register(
        id="AirHockey2D-v0",
        entry_point=f"{__name__}:AirHockey2DEnv",
        max_episode_steps=1500,
    )


register()
