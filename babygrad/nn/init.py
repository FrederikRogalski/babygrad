import math
from babygrad.tensor import Tensor

def xavier_uniform(tensor: Tensor):
    """Xavier uniform initialization otherwise known as Glorot initialization.
    Often used with zero-centered activation functions."""
    fan_in, fan_out = tensor.shape
    std = math.sqrt(6 / (fan_in + fan_out))
    tensor.data = tensor.data.uniform(-std, std, tensor.shape)
def kaiming_uniform(tensor: Tensor, a=0, mode='fan_in'):
    """Kaiming uniform initialization otherwise known as He initialization.
    Often used with ReLU activation functions."""
    match mode:
        case 'fan_in':
            fan =  tensor.shape[0]
        case 'fan_out':
            fan = tensor.shape[1]
        case 'fan_avg':
            fan = (tensor.shape[0] + tensor.shape[1]) / 2
        case _:
            raise ValueError(f"Invalid mode: {mode}")
    bound = math.sqrt(6 / (1 + a ** 2) / fan)
    tensor.data = tensor.data.uniform(-bound, bound, tensor.shape)
def xavier_normal(tensor: Tensor):
    """Xavier normal initialization otherwise known as Glorot initialization.
    Often used with zero-centered activation functions."""
    fan_in, fan_out = tensor.shape
    std = math.sqrt(2 / (fan_in + fan_out))
    tensor.data = tensor.data.normal(0, std, tensor.shape)
def kaiming_normal(tensor: Tensor, a=0, mode='fan_in'):
    """Kaiming normal initialization otherwise known as He initialization.
    Often used with ReLU activation functions."""
    match mode:
        case 'fan_in':
            fan =  tensor.shape[0]
        case 'fan_out':
            fan = tensor.shape[1]
        case 'fan_avg':
            fan = (tensor.shape[0] + tensor.shape[1]) / 2
        case _:
            raise ValueError(f"Invalid mode: {mode}")
    std = math.sqrt(2 / (1 + a ** 2) / fan)
    tensor.data = tensor.data.normal(0, std, tensor.shape)