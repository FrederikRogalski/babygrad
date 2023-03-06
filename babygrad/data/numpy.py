import numpy as np
from babygrad.data import Data

class NumpyData(np.ndarray, Data):
    def __new__(cls, data):
        return np.array(data, dtype=np.float32).view(cls)
    def zero(self):
        return np.zeros_like(self)
    def one(self):
        return np.ones_like(self)
    def exp(self):
        return np.exp(self)
    def log(self):
        return np.log(self)
    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)
    @staticmethod
    def ones(shape):
        return np.ones(shape, dtype=np.float32)
    @staticmethod
    def uniform(low, high, shape):
        return np.random.uniform(low, high, shape)
