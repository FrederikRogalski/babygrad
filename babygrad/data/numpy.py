import numpy as np
from babygrad.data import Data

class NumpyData(np.ndarray, Data):
    def __new__(cls, data):
        return np.array(data, dtype=np.float32).view(cls)
    def zero(self):
        return np.zeros_like(self).view(NumpyData)
    def one(self):
        return np.ones_like(self).view(NumpyData)
    def exp(self):
        return np.exp(self).view(NumpyData)
    def log(self):
        return np.log(self).view(NumpyData)

    def permute(self, dims):
        return np.transpose(super(), axes=dims).view(NumpyData)
    def expand(self, shape):
        return np.broadcast_to(super(), shape=shape).view(NumpyData)
    def reshape(self, shape):
        return np.reshape(super(), newshape=shape).view(NumpyData)
    def sum(self, dims):
        return np.sum(super(), axis=dims, keepdims=True).view(NumpyData)
    @staticmethod
    def zeros(shape):
        return np.zeros(shape).view(NumpyData)
    @staticmethod
    def ones(shape):
        return np.ones(shape).view(NumpyData)