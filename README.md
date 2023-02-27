# Babygrad
A baby sized deep learning framework 👼🏻🔢

## Usage

```python
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

graph(loss, globs = globals()).render("out/test", view=True, format="png")
```

<!-- Outputs the following picture -->
![test](https://user-images.githubusercontent.com/31591562/221460160-0ac981d3-3786-43f9-9038-ab54532670a5.png)
