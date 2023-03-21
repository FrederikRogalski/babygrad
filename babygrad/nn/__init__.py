import babygrad
from babygrad.tensor import Tensor
from babygrad.nn import init
from abc import ABC, abstractmethod

# Module is the base class for classes that want to implement a neural network module.
class Module(ABC):
    def __init__(self):
        self.parameters = []
    
    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
    
    @abstractmethod
    def forward(self, x):
        pass

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Tensor(init.xavier_uniform((in_features, out_features)))
        self.bias = babygrad.zeros(out_features)
        