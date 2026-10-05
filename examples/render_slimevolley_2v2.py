"""Render a 2v2 Slime Volleyball game in which all four players are the built-in AI.

Example command: `python render_slimevolley_2v2.py --out slimevolley_2v2_builtin.gif`
"""
import argparse

import jax
import jax.numpy as jnp
from PIL import Image

from evojax.task import slimevolley_2v2 as sv2


def main(config):
    state = sv2.init_game_state(jax.random.PRNGKey(config.seed), action_flag=(0, 0, 0, 0))
    step = jax.jit(sv2.step_game)
    key, frames = jax.random.PRNGKey(config.seed + 1), []
    for t in range(config.max_steps):
        key, k = jax.random.split(key)
        state, reward, _ = step(state, jnp.zeros((4, 3)), k)
        if t % config.every == 0:
            frames.append(Image.fromarray(sv2.render(state)))
        if sv2.detect_done(state):
            break
    frames[0].save(config.out, save_all=True, append_images=frames[1:], duration=40 * config.every, loop=0)
    print(f"{len(frames)} frames, {t + 1} steps, lives {state.agents.life.tolist()} -> {config.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--every", type=int, default=2, help="keep every n-th frame")
    parser.add_argument("--out", default="slimevolley_2v2_builtin.gif")
    main(parser.parse_args())
