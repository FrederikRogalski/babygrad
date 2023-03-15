import math
from babygrad.data import Data

class FloatData(float, Data):
    def item(self):
        return self
    def zero(self):
        return FloatData(0.0)
    def one(self):
        return FloatData(1.0)
    def exp(self):
        return FloatData(math.exp(self))
    def log(self):
        return FloatData(math.log(self))
    @property
    def shape(self):
        """Shape of the float is always empty"""
        return ()
    def permute(self, *dims):
        """Permute the shape of the float"""
        return FloatData(self)