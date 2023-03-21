import random
from babygrad.tensor import Tensor
from babygrad.visualize import graph

class Neuron:
    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1.0, 1.0), requires_grad=True) for _ in range(nin)]
        self.b = Tensor(random.uniform(-1.0, 1.0), requires_grad=True)
    
    def __call__(self, x):
        return sum((x*w for x,w in zip(x, self.w)), self.b).tanh()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, n):
        self.neurons = [Neuron(nin) for _ in range(n)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, layers: list[int] = (1,2,1)):
        self.layers = [Layer(layers[i], layers[i+1]) for i in range(len(layers)-1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if len(x) == 1: x = x[0]
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

insize = 3
l = MLP((insize, 1 , 1))
X = [[2.0, 3.0, -1.0],
     [3.0, -1.0, 0.5],
     [0.5, 1.0, 1.0],
     [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

y_hat = [l(x) for x in X]

loss = sum((y-y_hat)**2 for y,y_hat in zip(ys, y_hat)) / len(ys)
params = {f"Param{i}": p for i, p in enumerate(l.parameters())}

for _ in range(100):
    loss.zero_grads()
    loss.backward()
    print(loss)
    for p in l.parameters():
        p.data -= 0.1 * p.grad
graph(loss).view(cleanup=True)