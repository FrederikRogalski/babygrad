import babygrad
from typing import Iterable
from babygrad.tensor import Tensor
from babygrad.nn import init
from abc import ABC, abstractmethod

class Module(ABC):
    """The base class for all neural network modules."""
    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass
    def parameters(self) -> Iterable[Tensor]:
        pass

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Tensor.zeros((out_features, in_features), requires_grad=True)
        self.bias = Tensor.zeros((1, out_features), requires_grad=True)
        init.xavier_uniform(self.weight)
    def __call__(self, x) -> Tensor:
        return x @ self.weight.T + self.bias
    def parameters(self) -> Iterable[Tensor]:
        yield self.weight
        yield self.bias

class ReLU(Module):
    def __init__(self):
        pass
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Module):
    def __init__(self):
        pass
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class MLP(Module):
    def __init__(self, dims: tuple[int, ...], activation=ReLU):
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            if activation == ReLU:
                init.kaiming_uniform(self.layers[i].weight)
        self.activation = activation()
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
    def parameters(self) -> Iterable[Tensor]:
        for layer in self.layers:
            yield from layer.parameters()