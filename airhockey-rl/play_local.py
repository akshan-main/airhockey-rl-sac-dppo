"""Play against the scripted attacker locally with pygame.

You control the BOTTOM paddle with your mouse (click + drag).
The scripted attacker controls the TOP paddle.

Usage:
    python play_local.py
"""
import numpy as np
import pygame

from airhockey.env import AirHockeyEnv
from airhockey.eval_sac import scripted_attacker
from airhockey.physics import PhysicsConfig

# ── Colors ───────────────────────────────────────────────────
BG = (30, 30, 40)
FIELD = (20, 80, 20)
LINE = (60, 120, 60)
WALL = (80, 80, 90)
PUCK = (220, 220, 40)
BOT_PAD = (60, 140, 255)
TOP_PAD = (255, 80, 80)
GOAL = (40, 40, 50)
TEXT = (200, 200, 200)

FPS = 50


def main():

    # Higher paddle speed/accel caps than training so the human paddle
    # feels responsive. The AI opponent uses the same caps (fair fight).
    pc = PhysicsConfig(
        max_paddle_speed=1400.0,   # 2x training
        max_paddle_accel=12000.0,  # ~2.7x training
    )
    env = AirHockeyEnv(physics_config=pc, seed=42)

    # The SAC policy is too passive to be a fun opponent. Use the
    # scripted attacker as the primary opponent — it plays like a
    # real air hockey player (charges puck, aims at goal, defends).
    env.opponent = lambda obs_top: scripted_attacker(obs_top)


    # ── Pygame setup ─────────────────────────────────────────
    W, H = int(pc.width), int(pc.height)
    SCALE = 1.5
    SW, SH = int(W * SCALE), int(H * SCALE)

    pygame.init()
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Air Hockey — you (blue) vs SAC (red)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20)

    obs, _ = env.reset()
    bot_score = 0
    top_score = 0
    dragging = False
    last_mx, last_my = 0.0, 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                last_mx, last_my = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

        # ── Human input: click + drag to move the paddle ─────
        # Paddle moves based on mouse MOVEMENT (delta) while held,
        # not absolute position. Click without moving = paddle stays
        # perfectly still. Fast swipe = hard hit.
        s = env.physics.state
        dt = pc.dt
        if dragging:
            mx, my = pygame.mouse.get_pos()
            # Mouse delta in physics units.
            dmx = (mx - last_mx) / SCALE
            dmy = (my - last_my) / SCALE
            last_mx, last_my = mx, my
            # Convert mouse velocity to desired paddle velocity.
            mouse_vx = dmx / dt
            mouse_vy = dmy / dt
            # Accelerate to match mouse velocity.
            ax = (mouse_vx - s.bot_vx) / (pc.max_paddle_accel * dt)
            ay = (mouse_vy - s.bot_vy) / (pc.max_paddle_accel * dt)
            ax = np.clip(ax, -1.0, 1.0)
            ay = np.clip(ay, -1.0, 1.0)
            action = np.array([ax, ay], dtype=np.float32)
        else:
            # Not clicking — brake to stop.
            ax = -s.bot_vx / (pc.max_paddle_accel * dt)
            ay = -s.bot_vy / (pc.max_paddle_accel * dt)
            ax = np.clip(ax, -1.0, 1.0)
            ay = np.clip(ay, -1.0, 1.0)
            action = np.array([ax, ay], dtype=np.float32)

        obs, reward, term, trunc, info = env.step(action)

        if info["event"] == "goal_bot":
            bot_score += 1
        elif info["event"] == "goal_top":
            top_score += 1

        if term or trunc:
            obs, _ = env.reset()

        # ── Draw ─────────────────────────────────────────────
        screen.fill(BG)
        s = env.physics.state

        # Field
        field_rect = pygame.Rect(
            int(pc.wall * SCALE), int(pc.wall * SCALE),
            int((pc.width - 2 * pc.wall) * SCALE),
            int((pc.height - 2 * pc.wall) * SCALE),
        )
        pygame.draw.rect(screen, FIELD, field_rect)

        # Center line
        pygame.draw.line(
            screen, LINE,
            (int(pc.wall * SCALE), int(pc.height / 2 * SCALE)),
            (int((pc.width - pc.wall) * SCALE), int(pc.height / 2 * SCALE)),
            2,
        )

        # Goals
        gx1 = int((pc.width - pc.goal_width) / 2 * SCALE)
        gx2 = int((pc.width + pc.goal_width) / 2 * SCALE)
        pygame.draw.line(screen, GOAL, (gx1, int(pc.wall * SCALE)), (gx2, int(pc.wall * SCALE)), 4)
        pygame.draw.line(
            screen, GOAL,
            (gx1, int((pc.height - pc.wall) * SCALE)),
            (gx2, int((pc.height - pc.wall) * SCALE)),
            4,
        )

        # Puck
        pygame.draw.circle(
            screen, PUCK,
            (int(s.puck_x * SCALE), int(s.puck_y * SCALE)),
            int(pc.puck_radius * SCALE),
        )

        # Bot paddle (you — blue)
        pygame.draw.circle(
            screen, BOT_PAD,
            (int(s.bot_x * SCALE), int(s.bot_y * SCALE)),
            int(pc.paddle_radius * SCALE),
        )

        # Top paddle (SAC — red)
        pygame.draw.circle(
            screen, TOP_PAD,
            (int(s.top_x * SCALE), int(s.top_y * SCALE)),
            int(pc.paddle_radius * SCALE),
        )

        # Score
        score_txt = font.render(
            f"You (blue) {bot_score}  -  {top_score} SAC (red)", True, TEXT
        )
        screen.blit(score_txt, (SW // 2 - score_txt.get_width() // 2, 5))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
