from argparse import Namespace

import jax
import numpy as np

from .rocksample import RockSample
from .spec import load_spec, load_pomdp
from .popgym_ext import get_popgym_env
from .wrappers import OneHotObservationWrapper, OneHotActionConcatWrapper

def get_env(args: Namespace,
            rand_state: np.random.RandomState = None,
            rand_key: jax.random.PRNGKey = None,
            **kwargs):
    # for cleanliness, check rocksample first
    # we should never have a pomdp file named rocksample anyway so this should be ok!
    if args.spec == 'rocksample':
        env = RockSample(rand_key=rand_key, **kwargs)
    else:
        # try to load one of our native envs
        try:
            env, _ = load_pomdp(args.spec, rand_key=rand_state, **kwargs)

            # TODO this isn't DRY but idk how to fix this until David's TODO is fixed
            # TODO: some features are already encoded in a one-hot manner.
            if args.feature_encoding == 'one_hot':
                env = OneHotObservationWrapper(env)

        except AttributeError:
            # try to load from popgym
            # validate input: we need a custom gamma for popgym args as they don't come with a gamma
            if args.gamma is None:
                raise AttributeError("Can't load non-native environments without passing in gamma!")
            try:
                env = get_popgym_env(args, rand_key=rand_state, **kwargs)
                
            except AttributeError:
                # don't have anything else implemented
                raise NotImplementedError
            
            # TODO this isn't DRY but idk how to fix this until David's TODO is fixed
            # TODO: some features are already encoded in a one-hot manner.
            if args.feature_encoding == 'one_hot':
                env = OneHotObservationWrapper(env)
            
        
    if args.action_cond == 'cat':
        env = OneHotActionConcatWrapper(env)

    return env
