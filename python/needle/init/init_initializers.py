import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    # BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    # END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    # BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    # END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    a = gain * math.sqrt(3 / fan_in)

    shape = kwargs.get("shape", None)
    if shape is None:
        shape = (fan_in, fan_out)

    return rand(*shape, low=-1, high=1) * a
    # END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain * math.sqrt(1 / fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
