import math
from babygrad.tensor import Tensor
from babygrad.data.numpy import NumpyData as Data


def xavier_uniform(shape) -> Data:
    """Xavier uniform initialization otherwise known as Glorot initialization."""
    fan_in, fan_out = shape
    std = math.sqrt(6 / (fan_in + fan_out))
    return Data.uniform(-std, std, shape)

def kaiming_uniform(shape, a=0, mode='fan_in') -> Data:
    """Kaiming uniform initialization otherwise known as He initialization."""
    match mode:
        case 'fan_in':
            fan =  shape[0]
        case 'fan_out':
            fan = shape[1]
        case 'fan_avg':
            fan = (shape[0] + shape[1]) / 2
        case _:
            raise ValueError(f"Invalid mode: {mode}")
    bound = math.sqrt(6 / (1 + a ** 2) / fan)
    return Data.uniform(-bound, bound, shape)

def xavier_normal(shape) -> Data:
    """Xavier normal initialization otherwise known as Glorot initialization."""
    fan_in, fan_out = shape
    std = math.sqrt(2 / (fan_in + fan_out))
    return Data.normal(0, std, shape)