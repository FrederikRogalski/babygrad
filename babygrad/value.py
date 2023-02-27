import math
import warnings
from babygrad import log
from numbers import Number



class Value:
    operands: list['Value']
    symbol: str = 'v'
    value: float
    requires_grad_operands: list[bool]
    def __init__(self, operands: list['Value'], requires_grad: bool = False):
        self.operands = operands if isinstance(operands, list) else [operands]
        self.value = None
        self.grad = 0.0
        self.requires_grad = requires_grad
    
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
    def __neg__(self):
        return Neg(self)
    
    def maximum(self, other):
        return self._binary_op(other, Maximum)
    
    def exp(self):
        return Exp(self)
    def log(self):
        return Log(self)
    
    # ****** Non atomic operations *******
    def sigmoid(self):
        return (1.0 / (1.0 + (-self).exp()))
    def relu(self):
        return self._binary_op(0, Maximum)
    def tanh(self):
        return 2.0 * ((2.0 * self).sigmoid()) - 1.0
    
    
    def __repr__(self):
        # classname followed by operands
        return f"{type(self).__name__}({self.operands})"
    def __str__(self):
        return f"{self.symbol}({', '.join([op.symbol for op in self.operands if isinstance(op, Value)])})"
    
    def forward(self):
        self.value = self._forward()
        return self.value
    
    def _forward(self):
        return self.operands[0]
    
    def backward(self):
        self.grad = 1
        # topo sort        
        seq: list[Value] = []
        marked: dict[Value, bool] = {}
        
        def topo_sort(v: Value) -> bool:
            if not isinstance(v, Value):
                return False
            if v in marked:
                # return True if the node was marked as having descendants that require grad
                log.debug(f"Found circle in computation graph of '{self}': '{v}' was already marked as {'requiring grad' if marked[v] else 'not requiring grad'}")
                return marked[v]
            v.requires_grad_operands: list[bool] = [topo_sort(op) for op in v.operands] 
            descendents_require_grad = any(v.requires_grad_operands)
            if descendents_require_grad:
                seq.append(v)
            marked[v] = descendents_require_grad or v.requires_grad
            log.debug(f"Marked '{v}' as {'requiring grad' if marked[v] else 'not requiring grad'}")
            return marked[v]
        
        topo_sort(self)
        seq.reverse()
        log.debug(f"Topological sort of the computation graph of '{self}': {seq}")
        for v in seq:
            v._backward()

    def _backward(self):
        pass
    
    def zero_grad(self):
        self.grad = 0.0
        for op in self.operands:
            if isinstance(op, Value):
                op.zero_grad()


# ********* Subclasses of Value representing the different computations *************

class Add(Value):
    symbol = "+"
    def _forward(self):
        return self.operands[0].forward() + self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad if self.requires_grad_operands[0] else 0
        self.operands[1].grad += self.grad if self.requires_grad_operands[1] else 0

class Mul(Value):
    symbol = "*"
    def _forward(self):
        return self.operands[0].forward() * self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad * self.operands[1].value if self.requires_grad_operands[0] else 0
        self.operands[1].grad += self.grad * self.operands[0].value if self.requires_grad_operands[1] else 0

class Div(Value):
    symbol = "/"
    def _forward(self):
        return self.operands[0].forward() / self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad / self.operands[1].value if self.requires_grad_operands[0] else 0
        self.operands[1].grad += -(self.grad * self.operands[0].value)/(self.operands[1].value**2) if self.requires_grad_operands[1] else 0

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
        self.base.grad += self.grad * self.exp.value * self.base.value**(self.exp.value-1) if self.requires_grad_operands[0] else 0
        try:
            self.exp.grad += self.grad * self.value * math.log(self.base.value) if self.requires_grad_operands[1] else 0
        except ValueError:
            warnings.warn(f"Logarithm of negative number {self.base.value} is not defined, therfore gradient of exponent is not defined. Try clipping the base to a positive value.")
            

class Sub(Value):
    symbol = "-"
    def _forward(self):
        return self.operands[0].forward() - self.operands[1].forward()
    def _backward(self):
        self.operands[0].grad += self.grad if self.requires_grad_operands[0] else 0
        self.operands[1].grad += -self.grad if self.requires_grad_operands[1] else 0

class Maximum(Value):
    symbol = "max"
    def _forward(self):
        return max(self.operands[0].forward(), self.operands[1].forward())
    def _backward(self):
        # derivative is 1 if the operand is the maximum, 0 otherwise
        max_operand_0 = self.value == self.operands[0].value
        self.operands[0].grad += self.grad * max_operand_0 if self.requires_grad_operands[0] else 0
        self.operands[1].grad += self.grad * (not max_operand_0) if self.requires_grad_operands[1] else 0


# ********* Unary operations *************
class Neg(Value):
    symbol = "-"
    def _forward(self):
        return -self.operands[0].forward()
    def _backward(self):
        self.operands[0].grad += -self.grad if self.requires_grad_operands[0] else 0

class Exp(Value):
    symbol = "exp"
    def _forward(self):
        return math.exp(self.operands[0].forward())
    def _backward(self):
        self.operands[0].grad += self.grad * self.value if self.requires_grad_operands[0] else 0

class Log(Value):
    symbol = "log"
    def _forward(self):
        return math.log(self.operands[0].forward())
    def _backward(self):
        self.operands[0].grad += self.grad / self.operands[0].value if self.requires_grad_operands[0] else 0