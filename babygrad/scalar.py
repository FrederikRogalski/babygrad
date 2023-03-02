from babygrad.data import Data

class FloatData(float, Data):
    def item(self):
        return self.data
    def zero(self):
        return 0.0
    def one(self):
        return 1.0