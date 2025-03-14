import numpy as np

from grl.utils import softmax, reverse_softmax

# mem_probs are shape AOM->M
# policy_probs are shape OM->(A*M)

# P(m' | o, a, m) = P(a, m' | o, m) / P(a | o, m)
#                 = P(a, m' | o, m) / sum_m' P(a, m' | o, m)
#
# P(m', a | o, m) = P(m' | o, a, m) * P(a | o, m)

A = 3
M = 2
O = 4
aug_policy_probs = softmax(np.random.normal(size=np.prod([A, O, M, M])).reshape([O, M, A*M]), axis=-1)

def deconstruct_aug_policy(aug_policy_probs):
    O, M, AM = aug_policy_probs.shape
    A = AM // M
    aug_policy_probs_omam = aug_policy_probs.reshape([O, M, A, M])
    action_policy_probs_oma1 = aug_policy_probs_omam.sum(-1, keepdims=1) # (O, M, A, 1)
                                                                         #     pr(^|*)
    action_policy_probs = action_policy_probs_oma1.squeeze(-1)
    assert np.allclose(action_policy_probs.sum(-1), 1)

    aug_policy_logits_omam = reverse_softmax(aug_policy_probs_omam)
    action_policy_logits_oma1 = reverse_softmax(action_policy_probs_oma1)
    mem_logits_omam = (aug_policy_logits_omam - action_policy_logits_oma1) # (O, M, A, M)
    mem_probs_omam = softmax(mem_logits_omam, -1) # (O, M, A, M)
                                              #        pr(^|*)

    mem_probs = np.moveaxis(mem_probs_omam, -2, 0) # (A, O, M, M)
    assert np.allclose(mem_probs.sum(-1), 1)

    mem_logits = reverse_softmax(mem_probs)
    return mem_logits, action_policy_probs

def construct_aug_policy(mem_logits, action_policy_probs):
    A, O, M, _ = mem_logits.shape
    mem_probs = softmax(mem_logits, axis=-1) # (A, O, M, M)
    mem_probs_omam = np.moveaxis(mem_probs, 0, -2) # (O, M, A, M)

    action_policy_probs_oma1 = action_policy_probs[..., None] # (O, M, A, 1)

    aug_policy_probs_omam = (mem_probs_omam * action_policy_probs_oma1)
    aug_policy_probs = aug_policy_probs_omam.reshape([O, M, A*M])
    assert np.allclose(aug_policy_probs.sum(-1), 1)

    return aug_policy_probs

mem_logits, action_policy_probs = deconstruct_aug_policy(aug_policy_probs)
aug_policy_probs_reconstructed = construct_aug_policy(mem_logits, action_policy_probs)
assert np.allclose(aug_policy_probs_reconstructed, aug_policy_probs)

#%%
if __name__ == "__main__":
    import numpy as np

    from grl.environment.spec import load_pomdp
    from grl.memory import get_memory
    from grl.utils.augmented_policy import construct_aug_policy, deconstruct_aug_policy


    env, info = load_pomdp('tmaze_5_two_thirds_up', memory_id='18')
    pi = info['Pi_phi'][0]
    mem_params = get_memory('18',
                            n_obs=env.observation_space.n,
                            n_actions=env.action_space.n,
                            n_mem_states=2)

    inp_aug_pi = np.expand_dims(pi, axis=1).repeat(mem_params.shape[-1], axis=1)

    aug_policy = construct_aug_policy(mem_params, inp_aug_pi)
    mem_logits_reconstr, deconstr_aug_pi = deconstruct_aug_policy(aug_policy)

    softmax(mem_logits_reconstr, -1).round(3)
    softmax(mem_params, -1).round(3)

    print()
