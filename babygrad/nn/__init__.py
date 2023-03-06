from babygrad.value import Value
from abc import ABC, abstractmethod


class Module(ABC):
    """Base class for all neural network modules."""
    def __init__(self):
        self.parameters = []
    
    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
    
    @abstractmethod
    def forward(self, x):
        pass

def glorot_uniform(shape):
    with Value.no_grad():
        return Value.uniform(-1, 1, shape) * (6 / Value.sum(shape))**0.5
