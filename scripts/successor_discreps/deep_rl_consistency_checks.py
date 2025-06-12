from pathlib import Path
from sr_discrepancy_testing import *
from run_entropy_minimisation import get_policy_performance
from grl.utils.math import reverse_softmax

if __name__ == "__main__":
    # load 
    ckpt = "parity_check_seed(2024)_time(20250430-144524)_6cefe1d6ab327e77d378ef90bcb81fcf"
    #ckpt = "tmaze_5_seed(2024)_time(20250430-144528)_18774a8131fd395bda7a249f75c54b37"
    file_path = Path(f"/home/peter/repos/grl/scripts/deep_rl_pomdps/{ckpt}/data.npz")
    data = np.load(file_path)

    T = data["T"]
    R = data["R"]
    p0 = data["p0"]
    phi = data["phi"]
    pi = data["pi"]
    gamma = data["gamma"]
    hangman = data["hangman"]
    value_function = data["value_function"]
    pseudo_value_function = data["pseudo_value_function"]
    obs_of = data["obs_of"]

    n_actions = T.shape[0]
    n_states, n_obs = phi.shape
    print(f"Number of actions: {n_actions}")
    print(f"Number of states: {n_states}")
    print(f"Number of observations: {n_obs}")

    a, b = calculate_sr_discrepancy_raw(n_actions, n_states, n_obs, phi, T, p0, gamma, pi)
    discrepancy = np.sum(np.square(a - b))
    print(f"Discrepancy: {discrepancy}")

    mdp = MDP(T, R, p0, gamma)
    pomdp = POMDP(mdp, phi)
    pi_params = reverse_softmax(pi)
    perf = get_policy_performance(pi_params, pomdp)
    print(f"Performance: {perf}")

    learned_value = p0 @ phi @ value_function
    print(f"Learned value: {learned_value}")
    print(pseudo_value_function.shape)
    _, n_mdp_obs = pseudo_value_function.shape
    obs_projector = np.zeros((n_obs, n_mdp_obs))
    obs_projector[np.arange(n_obs), obs_of] = 1

    learned = pseudo_value_function[:3, :6]  # obs 6 is never seen in training; it is the last one. remaining entries are actions and reward, not actual observations
    true = (a@obs_projector)[:3, :6]
    #true = (b@obs_projector)[:3, :6]

    rel_error = np.abs(learned - true) / np.maximum(np.abs(true), 0.1)
    abs_error = np.abs(learned - true)
    max_rel_error = np.max(rel_error)
    avg_rel_error = np.mean(rel_error)
    max_error = np.max(abs_error)
    avg_error = np.mean(abs_error)
    print(f"Avg rel error: {100*avg_rel_error:.1f}%")
    print(f"Max rel error: {100*max_rel_error:.1f}%")
    print(f"Avg abs error: {avg_error:.3f}")
    print(f"Max abs error: {max_error:.3f}")
