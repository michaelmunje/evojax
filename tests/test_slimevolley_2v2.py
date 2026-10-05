"""Checks of the 2v2 Slime Volleyball rules (evojax/task/slimevolley_2v2.py)."""
import jax
import jax.numpy as jnp
import numpy as np

import evojax.task.slimevolley as sv
from evojax.task import slimevolley_2v2 as sv2


def play(steps=3000, seed=0):
    """All four players are built-in AIs; returns the states and rewards."""
    state = sv2.init_game_state(jax.random.PRNGKey(seed), action_flag=(0, 0, 0, 0))

    def step(state, key):
        state, reward, obs = sv2.step_game(state, jnp.zeros((4, 3)), key)
        return state, (state, reward)
    _, (states, rewards) = jax.lax.scan(step, state, jax.random.split(jax.random.PRNGKey(seed + 1), steps))
    return states, rewards


def test_observations_mirror_between_teams():
    state = sv2.init_game_state(jax.random.PRNGKey(0))
    state = state.replace(ball=state.ball.replace(vx=jnp.float32(0)))
    obs = sv2.get_obs(state)
    assert obs.shape == (4, sv2.OBS_SIZE)
    np.testing.assert_allclose(obs[0], obs[2])
    np.testing.assert_allclose(obs[1], obs[3])


def test_players_stay_in_their_half_and_teammates_never_overlap():
    states, _ = play()
    x, y, r = states.agents.x, states.agents.y, states.agents.r
    side = np.asarray(x * sv2.DIRECTION)
    assert side.min() >= sv.REF_WALL_WIDTH / 2 + 1.5 - 1e-4 and side.max() <= sv.REF_W / 2 - 1.5 + 1e-4
    for i, j in (sv2.LEFT, sv2.RIGHT):
        distance = np.hypot(np.asarray(x[:, i] - x[:, j]), np.asarray(y[:, i] - y[:, j]))
        assert distance.min() >= 3.0 - 1e-3


def test_points_are_zero_sum_team_lives():
    states, rewards = play()
    rewards = np.asarray(rewards)
    lives = np.asarray(states.agents.life)
    assert set(np.unique(rewards)) <= {-1.0, 0.0, 1.0} and np.abs(rewards).sum() > 0
    assert (lives[:, 0] == lives[:, 1]).all() and (lives[:, 2] == lives[:, 3]).all()  # lives are per team
    # each point costs the losing team one life (until a team has none left)
    first_out = np.argmax((lives <= 0).any(1)) if (lives <= 0).any() else len(lives) - 1
    lost_left = sv.MAXLIVES - lives[first_out, 0]
    lost_right = sv.MAXLIVES - lives[first_out, 2]
    assert lost_left == (rewards[:first_out + 1] > 0).sum() and lost_right == (rewards[:first_out + 1] < 0).sum()


def test_external_actions_control_only_flagged_players():
    state = sv2.init_game_state(jax.random.PRNGKey(0), action_flag=(0, 0, 1, 0))
    backward = jnp.zeros((4, 3)).at[2, 1].set(1.0)  # player 2 presses backward (away from the net)
    for k in range(10):
        state, _, _ = sv2.step_game(state, backward, jax.random.PRNGKey(k))
    assert float(state.agents.x[2]) > 8.0
    np.testing.assert_allclose(state.action[2], [0, 1, 0])


def test_vectorized_task_runs_with_evojax_policy():
    from evojax.policy.mlp import MLPPolicy
    task = sv2.SlimeVolley2v2(max_steps=50, test=True)
    policy = MLPPolicy(input_dim=sv2.OBS_SIZE, hidden_dims=[8], output_dim=3, output_act_fn="tanh")
    state = task.reset(jax.random.split(jax.random.PRNGKey(0), 3))
    params = jnp.zeros((3, policy.num_params))
    p_state = policy.reset(state)
    for _ in range(50):
        action, p_state = policy.get_actions(state, params, p_state)
        state, reward, done = task.step(state, action)
    assert state.obs.shape == (3, sv2.OBS_SIZE) and bool(done.all())
    assert sv2.SlimeVolley2v2.render(state).size == (sv.PIXEL_WIDTH, sv.PIXEL_HEIGHT)
