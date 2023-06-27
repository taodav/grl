from argparse import Namespace

from numpy import random

from .spec import load_spec, load_pomdp
from .popgym_ext import get_popgym_env
from .wrappers import OneHotObservationWrapper, OneHotActionConcatWrapper

def get_env(args: Namespace, rand_key: random.RandomState = None, **kwargs):
    # First we check our POMDP specs
    try:
        env, _ = load_pomdp(args.spec, rand_key=rand_key, **kwargs)
    except NotImplementedError:
        # try to load from popgym
        # validate input: we need a custom gamma for popgym args as they don't come with a gamma
        if args.gamma is None:
            raise AttributeError("Can't load non-native environments without passing in gamma!")
        try:
            env = get_popgym_env(args, rand_key=rand_key, **kwargs)
        except AttributeError:
            # don't have anything else implemented
            raise NotImplementedError

    if args.feature_encoding == 'one_hot':
        env = OneHotObservationWrapper(env)

    if args.action_cond == 'cat':
        env = OneHotActionConcatWrapper(env)

    return env
