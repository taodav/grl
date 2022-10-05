from pprint import pformat

import numpy as np

RTOL = 1e-3
ATOL = 1e-8

def pformat_vals(vals):
    """
    :param vals: dict
    """

    for k in vals.keys():
        vals[k] = np.array(vals[k])

    return pformat(vals)
