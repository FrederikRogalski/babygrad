from babygrad.value import Value
from babygrad.visualize import graph

def f(x):
    return x**2 + x

def mse(y, y_hat):
    return (y-y_hat)**2

x = Value(9)
y = Value(100)

y_hat = f(x)

loss = mse(y, y_hat)
loss.forward()
loss.backward()

graph(loss, globs = globals()).save("out/graph.gv")