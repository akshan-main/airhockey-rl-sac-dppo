# AirHockey2D-v0

```{figure} ../_static/img/envs/airhockey2d.png
:name: airhockey2d
```

|                       |                                                                              |
|-----------------------|------------------------------------------------------------------------------|
| Action Space          | `Box(-1.0, 1.0, (2,), float32)`                                              |
| Observation Space     | `Box(-1.0, 1.0, (10,), float32)`                                             |
| Import                | `gymnasium.make("AirHockey2D-v0")`                                            |

## Description

A 2D top-down air hockey environment. The agent controls the bottom paddle
and plays against a configurable opponent driving the top paddle. The puck
moves under linear drag with elastic wall reflection, and paddle/puck
contact uses line-of-impact elastic collision with mass ratio 4:1.

This is a simple, fully-observed continuous-control benchmark intended
for imitation learning, behavior cloning, and online RL methods on a small
environment that runs at >5,000 FPS on a single CPU thread.

## Action Space

The action is a `Box(-1.0, 1.0, (2,), float32)` representing the bottom
paddle's acceleration in (x, y), normalized so that the maximum
controllable acceleration in any direction is 4500 px/s².

| Index | Action       | Min  | Max  |
|-------|--------------|------|------|
| 0     | accel_x      | -1.0 | +1.0 |
| 1     | accel_y      | -1.0 | +1.0 |

## Observation Space

The observation is a 10-D `Box` with all values normalized to [0, 1] or [-1, 1]:

| Index | Observation                            | Min  | Max  |
|-------|----------------------------------------|------|------|
| 0     | puck position x / width                 | 0    | 1    |
| 1     | puck position y / height                | 0    | 1    |
| 2     | puck velocity x / max speed             | -1   | 1    |
| 3     | puck velocity y / max speed             | -1   | 1    |
| 4     | bot paddle x / width                    | 0    | 1    |
| 5     | bot paddle y / height                   | 0    | 1    |
| 6     | bot paddle vx / max speed               | -1   | 1    |
| 7     | bot paddle vy / max speed               | -1   | 1    |
| 8     | top paddle x / width                    | 0    | 1    |
| 9     | top paddle y / height                   | 0    | 1    |

## Rewards

| Event              | Reward |
|--------------------|--------|
| Score on opponent  | +10    |
| Concede goal       | -10    |
| Puck contact       | +0.1   |
| Per-step shaping   | 0.01 · (puck velocity toward opponent goal, normalized) |

## Starting State

The puck starts on the bottom paddle's side at (W/2, H * 0.72) with zero
velocity. Both paddles start at center-x near their defense lines.

## Episode End

- **Terminated**: a goal is scored or conceded.
- **Truncated**: episode length reaches 1500 steps.

## Arguments

```python
gymnasium.make(
    "AirHockey2D-v0",
    opponent=None,            # callable(obs_top) -> action; default no-op
    max_episode_steps=1500,
    reward_score=10.0,
    reward_concede=-10.0,
    reward_hit=0.1,
    reward_shaping_coef=0.01,
)
```

## Version History

- v0: Initial release. Two paddles, deterministic physics, no rendering
  (rendering layer is in the upstream repo).

## References

- Source: [github.com/akshan-main/airhockey-rl](https://github.com/akshan-main/airhockey-rl)
- Demonstrations: [huggingface.co/datasets/akshan-main/airhockey-demos](https://huggingface.co/datasets/akshan-main/airhockey-demos)
- Trained baseline: [huggingface.co/akshan-main/airhockey-dppo](https://huggingface.co/akshan-main/airhockey-dppo)
