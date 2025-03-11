import numpy as np
import jax
from jax.config import config

from grl.environment import load_pomdp
from grl.loss import discrep_loss, value_error
from grl.utils.policy_eval import analytical_pe

#%%

np.set_printoptions(precision=8)

spec = 'example_16_terminal'
seed = 42

np.set_printoptions(precision=8, suppress=True)
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

rand_key = None
np.random.seed(seed)
rand_key = jax.random.PRNGKey(seed)

pomdp, pi_dict = load_pomdp(spec, rand_key)
pomdp.gamma = 0.99999

if 'Pi_phi' in pi_dict and pi_dict['Pi_phi'] is not None:
    pi_phi = pi_dict['Pi_phi'][0]
    print(f'Pi_phi:\n {pi_phi}')

p = 0.5
pi_phi = np.array(
    [[p, 1-p],
     [p, 1-p]]
)

state_vals, mc_vals, td_vals, info = analytical_pe(pi_phi, pomdp)
state_vals['v']
state_vals['q']
mc_vals['v']
mc_vals['q']
td_vals['v']
td_vals['q']

discrep_loss(pi_phi, pomdp, alpha=0)

value_error(pi_phi, pomdp)
