# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2v2 Slime Volleyball: two teams of two slimes.

The physics, the built-in AI and the drawing code are those of the 1v1 task
(evojax/task/slimevolley.py), applied to four players. Rules:

  * Players 0 and 1 form the left team, players 2 and 3 the right team; players 0
    and 2 start near the net, 1 and 3 at the back, so player i mirrors player i + 2.
    Every player stays in its team's half, as in 1v1.
  * The ball bounces off all four players (in index order), the fence and the walls.
  * Teammates cannot overlap: after each move, overlapping teammates are moved
    apart horizontally (symmetrically, by the least distance, staying in their half). Opponents never meet (the fence
    separates them), so spacing on the own half is the new coordination problem.
  * Lives (5) and rewards are per team: +1 when the ball lands on the left half,
    -1 when it lands on the right half (from the right team's perspective).
  * Serves are drawn as in 1v1, except that vx and vy are independent random
    numbers (slimevolley.get_random_ball_v reads index 2 of a length-2 array,
    which JAX clamps to index 1, so the 1v1 serve has vx and vy correlated).

Observation of each player (20 values, divided by 10 as in 1v1), in its own
side's frame (its half is always x > 0):
    self (x, y, vx, vy), ball (4), opponent 0 (4), opponent 1 (4), teammate (4)
The first 12 values have the 1v1 layout (self, ball, an opponent), and the first
8 are what the built-in AI reads, so the 1v1 built-in AI plays any of the four
slots. Opponents are listed in index order.

Two interfaces:
  * Game-level functions for multi-agent use (e.g. ad hoc teamwork):
    init_game_state(key), step_game(state, buttons[4, 3], key), get_obs(state),
    detect_done(state), render(state). A player whose `action_flag` is 0 is
    played by the built-in AI.
  * SlimeVolley2v2, an EvoJAX VectorizedTask: the policy controls player 2 (right
    team) and plays with a built-in AI teammate against two built-in AIs.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from jax import random
from PIL import Image

import evojax.task.slimevolley as sv
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

N_PLAYERS = 4
LEFT, RIGHT = (0, 1), (2, 3)
DIRECTION = jnp.array([-1, -1, 1, 1])
START_X = jnp.array([-8.0, -16.0, 8.0, 16.0])  # players 0 and 2 start near the net, 1 and 3 at the back
OBS_SIZE = 20
# One hue per player: each team keeps its 1v1 colour for player 0 / 2, the teammate gets a clearly different one.
PLAYER_COLORS = [(0, 87, 184), (0, 190, 150), (254, 221, 0), (235, 80, 170)]  # blue, teal | yellow, pink
DAY_PLAYER_COLORS = [(240, 75, 0), (150, 60, 200), (0, 150, 255), (40, 170, 70)]


@dataclass
class GameState(object):
    ball: sv.ParticleState
    agents: sv.AgentState       # every field has a leading axis of N_PLAYERS
    hidden: jnp.ndarray         # (N_PLAYERS, 7) built-in AI recurrent states
    action_flag: jnp.ndarray    # (N_PLAYERS,) 1: use `action`, 0: built-in AI
    action: jnp.ndarray         # (N_PLAYERS, 3) buttons: forward, backward, jump


@dataclass
class State(TaskState):
    game_state: GameState
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


def random_ball_v(key):
    u = random.uniform(key, shape=(2,)) * 2 - 1
    return u[0] * 20, u[1] * 7.5 + 17.5


def new_ball(key):
    vx, vy = random_ball_v(key)
    return sv.initParticleState(0, sv.REF_W / 4, vx, vy, 0.5)


def init_game_state(key, action_flag=(0, 0, 1, 0)):
    """action_flag: which players take external actions (default: player 2)."""
    zeros = jnp.zeros(N_PLAYERS)
    agents = sv.AgentState(direction=DIRECTION, x=START_X, y=jnp.full(N_PLAYERS, sv.REF_U), r=jnp.full(N_PLAYERS, 1.5),
                           vx=zeros, vy=zeros, desired_vx=zeros, desired_vy=zeros,
                           life=jnp.full(N_PLAYERS, sv.MAXLIVES, jnp.int32))
    return GameState(ball=new_ball(key), agents=agents, hidden=jnp.zeros((N_PLAYERS, 7)),
                     action_flag=jnp.array(action_flag, jnp.int32),
                     action=jnp.zeros((N_PLAYERS, 3)))


def player(agents, i):
    return jax.tree.map(lambda x: x[i], agents)


def get_obs(game_state):
    """(N_PLAYERS, 20): self, ball, opponent 0, opponent 1, teammate, each in the player's own frame."""
    a, b = game_state.agents, game_state.ball

    def entity(x, y, vx, vy, d):
        return jnp.array([x * d, y, vx * d, vy])

    rows = []
    for i in range(N_PLAYERS):
        d = DIRECTION[i]
        team, other = (LEFT, RIGHT) if i in LEFT else (RIGHT, LEFT)
        mate = team[1] if team[0] == i else team[0]
        parts = [entity(a.x[i], a.y[i], a.vx[i], a.vy[i], d), entity(b.x, b.y, b.vx, b.vy, d)]
        parts += [entity(a.x[j], a.y[j], a.vx[j], a.vy[j], -d) for j in other]  # opponents, as in 1v1
        parts.append(entity(a.x[mate], a.y[mate], a.vx[mate], a.vy[mate], d))
        rows.append(jnp.concatenate(parts))
    return jnp.stack(rows) / 10.0


def _agent_step(agent, buttons):
    a = sv.Agent(agent, None)
    a.setAction(buttons)
    a.update()
    return a.p


def _lose_life(agent, result):
    a = sv.Agent(agent, None)
    a.updateLife(result)
    return a.p


def _separate(agents, i, j):
    """Teammates are discs that cannot overlap: if they do, move them apart horizontally, symmetrically and by
    the least distance that removes the overlap, while keeping both inside their half."""
    xi, xj, dy = agents.x[i], agents.x[j], agents.y[j] - agents.y[i]
    reach = agents.r[i] + agents.r[j]
    gap = jnp.maximum(jnp.abs(xj - xi), jnp.sqrt(jnp.maximum(reach * reach - dy * dy, 0.0)))
    mid, sign = (xi + xj) / 2, jnp.where(xj >= xi, 1.0, -1.0)
    inner, outer = sv.REF_WALL_WIDTH / 2 + agents.r[i], sv.REF_W / 2 - agents.r[i]
    low, high = (-outer, -inner) if i in LEFT else (inner, outer)
    shift = jnp.maximum(low - (mid - gap / 2), 0.0) - jnp.maximum(mid + gap / 2 - high, 0.0)
    x = agents.x.at[i].set(mid - sign * gap / 2 + shift).at[j].set(mid + sign * gap / 2 + shift)
    return agents.replace(x=x)


def step_game(game_state, buttons, key):
    """buttons: (N_PLAYERS, 3) used for players with action_flag 1. Returns the next state (with a new serve
    after a point), the reward for the right team and the observations."""
    obs = get_obs(game_state)
    hidden, builtin = jax.vmap(sv.baselinePolicy, in_axes=(0, 0, None))(
        obs, game_state.hidden, sv.initBaselinePolicyParams())
    flag = game_state.action_flag[:, None]
    actions = jnp.where(flag, buttons, builtin)
    agents = jax.vmap(_agent_step)(game_state.agents, actions)
    for i, j in (LEFT, RIGHT):
        agents = _separate(agents, i, j)

    ball = sv.Particle(game_state.ball, None)
    ball.applyAcceleration(0, sv.GRAVITY)
    ball.limitSpeed(sv.MAX_BALL_SPEED)
    ball.move()
    for i in range(N_PLAYERS):
        ball.bounceIfColliding(player(agents, i))
    ball.bounceIfColliding(sv.initParticleState(0, sv.REF_WALL_HEIGHT, 0, 0, sv.REF_WALL_WIDTH / 2))
    reward = -ball.checkEdges()  # +1: the ball landed on the left half
    agents = jax.vmap(_lose_life, in_axes=(0, None))(agents, reward)

    served = new_ball(key)
    ball = jax.tree.map(lambda old, new: jnp.where(reward == 0, old, new), ball.p, served)
    state = GameState(ball, agents, hidden, game_state.action_flag, actions)
    return state, reward, get_obs(state)


def detect_done(game_state):
    return jnp.any(game_state.agents.life <= 0)


def render(game_state):
    """An RGB image (numpy) of the game, as in 1v1, with one colour per player (PLAYER_COLORS)."""
    game = sv.Game(sv.initGameState(0.0, 0.0))
    canvas = sv.create_canvas(c=sv.BACKGROUND_COLOR)
    canvas = game.fence.display(canvas)
    canvas = game.fenceStub.display(canvas)
    ball = game_state.ball
    colors = PLAYER_COLORS if sv.BACKGROUND_COLOR == (11, 16, 19) else DAY_PLAYER_COLORS
    for i in range(N_PLAYERS):
        canvas = sv.Agent(player(game_state.agents, i), colors[i]).display(canvas, ball.x, ball.y)
    canvas = sv.Particle(ball, sv.BALL_COLOR).display(canvas)
    canvas = game.ground.display(canvas)
    return sv.downsize_image(canvas)


class SlimeVolley2v2(VectorizedTask):
    """The policy controls player 2 (right team) with a built-in AI teammate (player 3) against two built-in
    AIs. Training plays all max_steps steps; testing ends when a team has lost all its lives."""

    def __init__(self, max_steps: int = 3000, test: bool = False):
        self.max_steps = max_steps
        self.obs_shape = tuple([OBS_SIZE, ])
        self.act_shape = tuple([3, ])
        self.test = test

        def reset_fn(key):
            next_key, key = random.split(key)
            game_state = init_game_state(key)
            return State(game_state=game_state, obs=get_obs(game_state)[2],
                         steps=jnp.zeros((), dtype=int), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            next_key, key = random.split(state.key)
            buttons = jnp.zeros((N_PLAYERS, 3)).at[2].set(action)
            game_state, reward, obs = step_game(state.game_state, buttons, key)
            steps = state.steps + 1
            done_test = jnp.bitwise_or(detect_done(game_state), steps >= max_steps)
            done = jnp.where(self.test, done_test, steps >= max_steps)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            return State(game_state=game_state, obs=obs[2], steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self, state: State, action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int = 0) -> Image:
        """Render a specified task."""
        return Image.fromarray(np.asarray(render(jax.tree.map(lambda x: x[task_id], state.game_state))))
