"""Air hockey physics.

Semi-implicit Euler with substepping, elastic line-of-impact paddle/puck
collisions, and wall reflection with restitution.

Coordinate system: (0,0) top-left, x rightward, y downward. Top paddle
defends y near 0, bottom paddle defends y near height.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass
class PhysicsConfig:
    width: float = 320.0
    height: float = 540.0
    wall: float = 12.0
    goal_width: float = 110.0
    puck_radius: float = 11.0
    paddle_radius: float = 19.0
    puck_mass: float = 1.0
    paddle_mass: float = 4.0
    e_wall: float = 0.94
    e_paddle: float = 0.96
    drag: float = 0.18  # linear damping per second
    max_puck_speed: float = 950.0
    max_paddle_speed: float = 700.0
    max_paddle_accel: float = 4500.0
    dt: float = 1.0 / 50.0
    substeps: int = 3


@dataclass
class State:
    """Mutable rigid-body state. All quantities in pixel units & seconds."""
    puck_x: float = 0.0
    puck_y: float = 0.0
    puck_vx: float = 0.0
    puck_vy: float = 0.0
    top_x: float = 0.0
    top_y: float = 0.0
    top_vx: float = 0.0
    top_vy: float = 0.0
    bot_x: float = 0.0
    bot_y: float = 0.0
    bot_vx: float = 0.0
    bot_vy: float = 0.0
    top_score: int = 0
    bot_score: int = 0
    last_event: str = ""  # "goal_top", "goal_bot", "hit_top", "hit_bot", ""


class AirHockeyPhysics:
    """Pure-NumPy air hockey simulator. No rendering; the env wraps it."""

    def __init__(self, config: Optional[PhysicsConfig] = None, seed: int = 0):
        self.cfg = config or PhysicsConfig()
        self.rng = np.random.default_rng(seed)
        self.state = State()
        self._derive_field()
        self.reset(serve_to="top")

    # ── Geometry ───────────────────────────────────────────────
    def _derive_field(self) -> None:
        c = self.cfg
        self.field_x1 = c.wall + c.puck_radius
        self.field_x2 = c.width - c.wall - c.puck_radius
        self.field_y1 = c.wall + c.puck_radius
        self.field_y2 = c.height - c.wall - c.puck_radius
        self.goal_x1 = (c.width - c.goal_width) / 2
        self.goal_x2 = (c.width + c.goal_width) / 2
        self.top_min_y = c.wall + c.paddle_radius
        self.top_max_y = c.height / 2 - c.paddle_radius
        self.bot_min_y = c.height / 2 + c.paddle_radius
        self.bot_max_y = c.height - c.wall - c.paddle_radius
        self.paddle_min_x = c.wall + c.paddle_radius
        self.paddle_max_x = c.width - c.wall - c.paddle_radius

    # ── Reset ──────────────────────────────────────────────────
    def reset(self, serve_to: Literal["top", "bot"] = "top") -> None:
        c = self.cfg
        s = self.state
        # Puck serves with a random x-offset so the agent can't exploit a
        # fixed starting trajectory.
        x_jitter = float(self.rng.uniform(-40.0, 40.0))
        s.puck_x = c.width / 2 + x_jitter
        s.puck_y = c.height * (0.30 if serve_to == "top" else 0.70)
        s.puck_vx = 0.0
        s.puck_vy = 0.0
        # Paddles reset to their home positions. The top paddle home is
        # well clear of the puck's serve trajectory so a straight shot
        # from the bot can't trivially bounce off a stationary top.
        s.top_x = c.width / 2
        s.top_y = c.wall + c.paddle_radius + 30
        s.top_vx = s.top_vy = 0.0
        s.bot_x = c.width / 2
        s.bot_y = c.height - c.wall - c.paddle_radius - 30
        s.bot_vx = s.bot_vy = 0.0
        s.last_event = ""

    def hard_reset(self, serve_to: Literal["top", "bot"] = "top") -> None:
        self.state.top_score = 0
        self.state.bot_score = 0
        self.reset(serve_to)

    def step(self, top_accel: np.ndarray, bot_accel: np.ndarray) -> str:
        """Advance one env step. Each accel is (ax, ay) in canvas units/s².
        Returns one of: "goal_top", "goal_bot", "hit_top", "hit_bot", "".
        Goals dominate hits: if a goal was scored in any substep, the
        return value is that goal, not a contact.
        """
        c = self.cfg
        sub_dt = c.dt / c.substeps
        self.state.last_event = ""
        step_event = ""

        for _ in range(c.substeps):
            self._integrate_paddle("top", top_accel, sub_dt)
            self._integrate_paddle("bot", bot_accel, sub_dt)
            self._integrate_puck(sub_dt)
            self._collide_paddle("top")
            self._collide_paddle("bot")

            if self.state.last_event:
                if self.state.last_event.startswith("goal"):
                    step_event = self.state.last_event
                    serve = "top" if step_event == "goal_bot" else "bot"
                    self.reset(serve_to=serve)
                    break
                elif not step_event.startswith("goal"):
                    step_event = self.state.last_event

        # reset() wipes last_event, so restore it before returning.
        self.state.last_event = step_event
        return step_event

    # ── Paddle integration (semi-implicit Euler + clamping) ────
    def _integrate_paddle(self, which: str, accel: np.ndarray, dt: float) -> None:
        c = self.cfg
        s = self.state
        ax, ay = float(accel[0]), float(accel[1])
        a_norm = float(np.hypot(ax, ay))
        if a_norm > c.max_paddle_accel:
            ax = ax / a_norm * c.max_paddle_accel
            ay = ay / a_norm * c.max_paddle_accel

        if which == "top":
            s.top_vx += ax * dt
            s.top_vy += ay * dt
            sp = float(np.hypot(s.top_vx, s.top_vy))
            if sp > c.max_paddle_speed:
                s.top_vx = s.top_vx / sp * c.max_paddle_speed
                s.top_vy = s.top_vy / sp * c.max_paddle_speed
            s.top_x += s.top_vx * dt
            s.top_y += s.top_vy * dt
            # Clamp position
            s.top_x = float(np.clip(s.top_x, self.paddle_min_x, self.paddle_max_x))
            new_y = float(np.clip(s.top_y, self.top_min_y, self.top_max_y))
            if new_y != s.top_y:
                s.top_vy = 0.0
                s.top_y = new_y
        else:
            s.bot_vx += ax * dt
            s.bot_vy += ay * dt
            sp = float(np.hypot(s.bot_vx, s.bot_vy))
            if sp > c.max_paddle_speed:
                s.bot_vx = s.bot_vx / sp * c.max_paddle_speed
                s.bot_vy = s.bot_vy / sp * c.max_paddle_speed
            s.bot_x += s.bot_vx * dt
            s.bot_y += s.bot_vy * dt
            s.bot_x = float(np.clip(s.bot_x, self.paddle_min_x, self.paddle_max_x))
            new_y = float(np.clip(s.bot_y, self.bot_min_y, self.bot_max_y))
            if new_y != s.bot_y:
                s.bot_vy = 0.0
                s.bot_y = new_y

    # ── Puck integration with wall collisions ──────────────────
    def _integrate_puck(self, dt: float) -> None:
        c = self.cfg
        s = self.state
        # Linear drag
        damp = max(0.0, 1.0 - c.drag * dt)
        s.puck_vx *= damp
        s.puck_vy *= damp
        # Speed cap
        sp = float(np.hypot(s.puck_vx, s.puck_vy))
        if sp > c.max_puck_speed:
            s.puck_vx = s.puck_vx / sp * c.max_puck_speed
            s.puck_vy = s.puck_vy / sp * c.max_puck_speed
        # Semi-implicit Euler position update
        s.puck_x += s.puck_vx * dt
        s.puck_y += s.puck_vy * dt
        # Wall collisions
        if s.puck_x < self.field_x1:
            s.puck_x = self.field_x1
            s.puck_vx = -s.puck_vx * c.e_wall
        elif s.puck_x > self.field_x2:
            s.puck_x = self.field_x2
            s.puck_vx = -s.puck_vx * c.e_wall
        # Top wall (with goal mouth)
        if s.puck_y < self.field_y1:
            if self.goal_x1 + c.puck_radius < s.puck_x < self.goal_x2 - c.puck_radius:
                s.bot_score += 1
                s.last_event = "goal_bot"
                return
            s.puck_y = self.field_y1
            s.puck_vy = -s.puck_vy * c.e_wall
        # Bottom wall
        if s.puck_y > self.field_y2:
            if self.goal_x1 + c.puck_radius < s.puck_x < self.goal_x2 - c.puck_radius:
                s.top_score += 1
                s.last_event = "goal_top"
                return
            s.puck_y = self.field_y2
            s.puck_vy = -s.puck_vy * c.e_wall

    # ── Paddle ↔ puck elastic collision ────────────────────────
    def _collide_paddle(self, which: str) -> None:
        c = self.cfg
        s = self.state
        if which == "top":
            px, py, pvx, pvy = s.top_x, s.top_y, s.top_vx, s.top_vy
        else:
            px, py, pvx, pvy = s.bot_x, s.bot_y, s.bot_vx, s.bot_vy

        dx = s.puck_x - px
        dy = s.puck_y - py
        dist = float(np.hypot(dx, dy))
        min_dist = c.puck_radius + c.paddle_radius
        if dist >= min_dist or dist < 1e-3:
            return

        nx = dx / dist
        ny = dy / dist
        v_rel_x = s.puck_vx - pvx
        v_rel_y = s.puck_vy - pvy
        v_rel_n = v_rel_x * nx + v_rel_y * ny
        if v_rel_n >= 0:
            return  # separating already

        e = c.e_paddle
        j = -(1 + e) * v_rel_n / (1 / c.puck_mass + 1 / c.paddle_mass)
        s.puck_vx += (j / c.puck_mass) * nx
        s.puck_vy += (j / c.puck_mass) * ny
        if which == "top":
            s.top_vx -= (j / c.paddle_mass) * nx
            s.top_vy -= (j / c.paddle_mass) * ny
        else:
            s.bot_vx -= (j / c.paddle_mass) * nx
            s.bot_vy -= (j / c.paddle_mass) * ny

        # Positional correction
        overlap = min_dist - dist
        s.puck_x += nx * overlap
        s.puck_y += ny * overlap
        s.last_event = f"hit_{which}"

    # ── Observation helpers ────────────────────────────────────
    def get_obs(self, perspective: Literal["top", "bot"] = "bot") -> np.ndarray:
        """10D observation in the perspective paddle's reference frame.
        Coordinates are mirrored vertically for the top paddle so the
        same policy can drive either side."""
        c = self.cfg
        s = self.state
        if perspective == "bot":
            # native frame
            obs = np.array([
                s.puck_x / c.width,
                s.puck_y / c.height,
                s.puck_vx / c.max_puck_speed,
                s.puck_vy / c.max_puck_speed,
                s.bot_x / c.width,
                s.bot_y / c.height,
                s.bot_vx / c.max_paddle_speed,
                s.bot_vy / c.max_paddle_speed,
                s.top_x / c.width,
                s.top_y / c.height,
            ], dtype=np.float32)
        else:
            # mirror y so top paddle "looks" like a bottom paddle
            obs = np.array([
                s.puck_x / c.width,
                1.0 - s.puck_y / c.height,
                s.puck_vx / c.max_puck_speed,
                -s.puck_vy / c.max_puck_speed,
                s.top_x / c.width,
                1.0 - s.top_y / c.height,
                s.top_vx / c.max_paddle_speed,
                -s.top_vy / c.max_paddle_speed,
                s.bot_x / c.width,
                1.0 - s.bot_y / c.height,
            ], dtype=np.float32)
        return obs
