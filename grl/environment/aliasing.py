from grl.mdp import POMDP


def map_to_strict_aliasing(pomdp: POMDP) -> POMDP:
    states_to_expand = {}
    for i, phi_row in enumerate(pomdp.phi):
        state_maps_to_n_obs = (phi_row > 0).sum()
        if state_maps_to_n_obs > 1:
            pass

