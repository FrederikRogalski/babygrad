import math
import logging
import warnings
from numbers import Number

logging.basicConfig(level=logging.INFO)

class Value:
    operands: list['Value']
    symbol: str = 'v'
    value: float
    def __init__(self, operands: list['Value']):
        self.operands = operands if isinstance(operands, list) else [operands]
        self.value = None
        self.grad = 0.0
    
    def _binary_op(self, operand, Op, swap=False):
        match operand:
            case Value():
                pass
            case Number():
                operand = Value(operand)
            case _:
                raise TypeError(f"unsupported operand type(s) for {Op.symbol}: '{type(self).__name__}' and '{type(operand).__name__}'")
        if swap:
            return Op([operand, self])
        return Op([self, operand])
            
    def __add__(self, addend: 'Value'):
        return self._binary_op(addend, Add)
    __radd__ = __add__
    def __mul__(self, multiplicant: 'Value'):
        return self._binary_op(multiplicant, Mul)
    __rmul__ = __mul__
    def __truediv__(self, divisor: 'Value'):
        return self._binary_op(divisor, Div)
    def __rtruediv__(self, dividend: 'Value'):
        return self._binary_op(dividend, Div, swap=True)
    def __pow__(self, exponent: 'Value'):
        return self._binary_op(exponent, Pow)
    def __rpow__(self, base: 'Value'):
        return self._binary_op(base, Pow, swap=True)
    def __sub__(self, subtrahend: 'Value'):
        return self._binary_op(subtrahend, Sub)
    def __rsub__(self, minuend: 'Value'):
        return self._binary_op(minuend, Sub, swap=True)
    
    def __repr__(self):
        # classname followed by operands
        return f"{type(self).__name__}({self.operands})"
    def __str__(self):
        return f"{self.symbol}({', '.join(map(str, self.operands))})"
    
    def forward(self):
        self.value = self._forward()
        return self.value
    
    def _forward(self):
        return self.operands[0]
    
    def backward(self):
        self.grad = 1
        # topo sort
        seq = []
        marked = set()
        def topo_sort(v: Value):
            if v in marked:
                logging.info(f"Cycle detected in the computation graph of '{self}' at '{v}'")
                return
            marked.add(v)
            for operand in v.operands:
                if isinstance(operand, Value):
                    topo_sort(operand)
            seq.append(v)
        topo_sort(self)
        seq.reverse()
        logging.info(f"Topological sort of the computation graph of '{self}': {seq}")
        for v in seq:
            v._backward()
        
        
    
    def _backward(self):
        pass


# ********* Subclasses of Value representing the different computations *************

class Add(Value):
    symbol = "+"
    def _forward(self):
        return self.operands[0].forward() + self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad
        self.operands[1].grad += self.grad

class Mul(Value):
    symbol = "*"
    def _forward(self):
        return self.operands[0].forward() * self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad * self.operands[1].value
        self.operands[1].grad += self.grad * self.operands[0].value

class Div(Value):
    symbol = "/"
    def _forward(self):
        return self.operands[0].forward() / self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad / self.operands[1].value
        self.operands[1].grad += -(self.grad * self.operands[0].value)/(self.operands[1].value**2)

class Pow(Value):
    symbol = "**"
    def _forward(self):
        return self.base.forward() ** self.exp.forward()
    @property
    def base(self):
        return self.operands[0]
    @property
    def exp(self):
        return self.operands[1]
    def _backward(self):
        self.base.grad += self.grad * self.exp.value * self.base.value**(self.exp.value-1)
        try:
            self.exp.grad += self.grad * self.value * math.log(self.base.value)
        except ValueError:
            warnings.warn(f"Logarithm of negative number {self.base.value} is not defined, therfore gradient of exponent is not defined. Try clipping the base to a positive value.")
            

class Sub(Value):
    symbol = "-"
    def _forward(self):
        return self.operands[0].forward() - self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad
        self.operands[1].grad += -self.grad