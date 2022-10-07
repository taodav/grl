import numpy as np

from grl import MDP, AbstractMDP, PolicyEval, environment
from grl.value_iteration import value_iteration

def test_vi():
    chain_length = 10
    spec = environment.load_spec('simple_chain', memory_id=None)

    print(f"Testing value iteration on Simple Chain.")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    ground_truth_vals = spec['gamma'] ** np.arange(chain_length - 2, -1, -1)
    v = value_iteration(mdp.T, mdp.R, mdp.gamma)

    print(f"Calculated values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert np.all(np.isclose(v[:-1], ground_truth_vals, atol=1e-2))
