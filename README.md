# Babygrad
A baby sized deep learning framework 👼🏻🔢

## Usage

```python
from babygrad.tensor import Tensor
from babygrad.visualize import graph

def f(x):
    return x**2 + x

def mse(y, y_hat):
    return (y-y_hat)**2

x = Tensor(9, requires_grad=True)
y = Tensor(100)

y_hat = f(x)

loss = mse(y, y_hat)
loss.backward()

graph(loss).render("out/test", view=True, format="png")
```

<!-- Outputs the following picture -->
![test](https://user-images.githubusercontent.com/31591562/226619448-62a2d7cf-4696-403e-8ed5-d001d7a222f5.png)
