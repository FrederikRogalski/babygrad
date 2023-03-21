import numpy as np
from abc import ABC, abstractmethod

class Data(ABC):
    @abstractmethod
    def __init__(self, data):
        pass
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...] | tuple[()]:
        pass
    @abstractmethod
    def item(self) -> float:
        """Checks if self contains a single element and returns it. If self has multiple elements it raises a ValueError"""
        pass
    # class requires Add, Mul, Divide, Pow
    @abstractmethod
    def __add__(self, other):
        pass
    @abstractmethod
    def __radd__(self, other):
        pass
    @abstractmethod
    def __sub__(self, other):
        pass
    @abstractmethod
    def __rsub__(self, other):
        pass
    @abstractmethod
    def __mul__(self, other):
        pass
    @abstractmethod
    def __rmul__(self, other):
        pass
    @abstractmethod
    def __truediv__(self, other):
        pass
    @abstractmethod
    def __rtruediv__(self, other):
        pass
    @abstractmethod
    def __pow__(self, other):
        pass
    @abstractmethod
    def __rpow__(self, other):
        pass
    @abstractmethod
    def __neg__(self):
        pass
    @abstractmethod
    def __gt__(self, other):
        pass
    @abstractmethod
    def __matmul__(self, other):
        pass
    
    @abstractmethod
    def exp(self):
        """Returns the exponential of self."""
    
    def permute(self, dims):
        """Returns a data instance with the dimensions permuted."""
    def expand(self, shape):
        """Returns a data instance with the dimensions expanded."""
    def reshape(self, shape):
        """Returns a reshaped data instance."""
    def sum(self, dims):
        """Returns the sum of self along the given dimensions."""
        
# ********** None atmoic operations **********
    
    def zero(self):
        """Returns a data instance of zeros of the same type and shape as self."""
        return self.zeros_like(self)
    def one(self):
        """Returns a data instance of ones of the same type and shape as self."""
        return self.ones_like(self)
    @classmethod
    def zeros(cls, shape):
        """Returns a tensor of zeros of the given tpye and shape"""
        return cls(0).reshape(shape=tuple(1 for _ in shape)).expand(shape=shape)
    @classmethod
    def ones(cls, shape):
        """Returns a tensor of ones of the given tpye and shape"""
        return cls(1).reshape(shape=tuple(1 for _ in shape)).expand(shape=shape)
    @classmethod
    def zeros_like(cls, data):
        """Returns a Tensor of zeros of the same type and shape as data."""
        return cls.zeros(data.shape)
    @classmethod
    def ones_like(cls, data):
        """Returns a Tensor of ones of the same type and shape as data."""
        return cls.ones(data.shape)
    @classmethod
    def uniform(cls, shape):
        """Returns a uniform random tensor of the given shape. With values between and including 0 and 1."""
        return cls(np.random.uniform(0, 1, shape))
    @classmethod
    def normal(cls, shape):
        """Returns a normal random tensor of the given shape. With mean 0 and standard deviation 1."""
        return cls(np.random.normal(0, 1, shape))