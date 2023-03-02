from babygrad.scalar import FloatData as Data
from abc import ABC, abstractmethod

class Operand(ABC):
    data: Data
    symbol: str
    def __init__(self, requires_grad = False):
        self.requires_grad = requires_grad
        # Thisvariable will be set to either True or False for each forward pass depending on whether any of the operands require a gradient
        # Therefore it indicates an indirect dependency on the gradient of a child node
        self._decendant_requires_grad = None
    
    def item(self):
        """Checks if self.data contains a single element and returns it."""
        return self.data.item()
    
    def __repr__(self):
        return f"{type(self).__name__}({self.data})"
    
    @abstractmethod
    def __call__(self):
        """Computes the whole computation graph and assigns self.data for each node."""
        pass
    
    def _binary_op(self, operand, Op, swap=False):
        match operand:
            case Operand():
                pass
            case _:
                operand = Value(data=operand)
        return Op([operand, self]) if swap else Op([self, operand])
    
    # ****** Atomic operations ********
    def __add__(self, addend):
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





class Value(Operand):
    symbol = "V"
    def __init__(self, data: Data, requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self.data = data if isinstance(data, Data) else Data(data)
    def __call__(self):
        return self.data




class Operator(Operand):
    symbol = "O"
    def __init__(self, operands: list[Operand], requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self.operands = operands
        self.data = self._forward()
    
    @classmethod
    def _dfs(cls, current: Operand, visited: set[Operand], seq: list['Operator']):
        if current in visited:
            return
        visited.add(current)
        if isinstance(current, Operator):
            for op in current.operands:
                cls._dfs(op, visited, seq)
            seq.append(current)
    
    def __call__(self):
        """Forward pass of the computation graph."""
        # create a topo sort of the computation graph
        self._topo_sorted_graph: list[Operator] = []
        self._dfs(self, set(), self._topo_sorted_graph)
        
        for operator in self._topo_sorted_graph:
            self._op_requires_grad = []
            for operand in operator.operands:
                oprg = operand.requires_grad or operand._decendant_requires_grad
                self._op_requires_grad.append(oprg)
                if oprg:
                    operand.grad = self.data.zero()
            self._decendant_requires_grad = any(self._op_requires_grad)
            operator.data = operator._forward()
    
    @abstractmethod
    def _forward(self) -> Data:
        """Computes only this node of the computation graph."""
    
    def backward(self):
        # Compute gradients in reverse topo sort order
        for operator in reversed(self._topo_sorted_graph):
            operand_gradients = operator._backward()
            # Add the weighted gradients to the child nodes
            for operand, operand_gradient in zip(operator.operands, operand_gradients):
                if operand_gradient is None:
                    continue
                operand.grad += operand_gradient * self.grad
    
    @abstractmethod
    def _backward(self) -> list[Data | None]:
        """Computes the gradients of this nodes operands if they are required."""
        

# ********* Atomic binary operations *************

class Add(Operator):
    symbol = "+"
    def _forward(self):
        return sum(op.data for op in self.operands)
    def _backward(self):
        return  self.operands[0].data.one() if self._op_requires_grad[0] else None, \
                self.operands[1].data.one() if self._op_requires_grad[1] else None
class Sub(Operator):
    symbol = "-"
    def _forward(self):
        return self.operands[0].data - sum(op.data for op in self.operands[1:])
    def _backward(self):
        return  self.operands[0].data.one() if self._op_requires_grad[0] else None, \
                -self.operands[1].data.one() if self._op_requires_grad[1] else None
class Mul(Operator):
    symbol = "*"
    def _forward(self):
        return self.operands[0].data * self.operands[1].data
    def _backward(self):
        return  self.operands[1].data if self._op_requires_grad[0] else None, \
                self.operands[0].data if self._op_requires_grad[1] else None
class Div(Operator):
    symbol = "/"
    def _forward(self):
        return self.operands[0].data / self.operands[1].data
    def _backward(self):
        return  (1.0 / self.operands[1].data) if self._op_requires_grad[0] else None, \
                -(self.operands[0].data / (self.operands[1].data ** 2)) if self._op_requires_grad[1] else None
class Pow(Operator):
    symbol = "**"
    def _forward(self):
        return self.operands[0].data ** self.operands[1].data
    def _backward(self):
        return  (self.operands[1].data * (self.operands[0].data ** (self.operands[1].data - 1))) if self._op_requires_grad[0] else None, \
                ((self.operands[0].data ** self.operands[1].data) * self.operands[0].data.log()) if self._op_requires_grad[1] else None
class Maximum(Operator):
    symbol = "max"
    def _forward(self):
        return max(self.operands[0].data, self.operands[1].data)
    def _backward(self):
        max_op = max(self.operands, key=lambda op: op.data)
        return  self.operands[0].data.one() if max_op == self.operands[0] else self.operands[0].data.zero() if self._op_requires_grad[0] else None, \
                self.operands[1].data.one() if max_op == self.operands[1] else self.operands[1].data.zero() if self._op_requires_grad[1] else None


# ********* Atomic unary operations *************

class Neg(Operator):
    symbol = "-"
    def _forward(self):
        return -self.operands[0].data
    def _backward(self):
        return -self.operands[0].data.one() if self._op_requires_grad[0] else None
class Exp(Operator):
    symbol = "exp"
    def _forward(self):
        return self.operands[0].data.exp()
    def _backward(self):
        return self.data if self._op_requires_grad[0] else None
class Log(Operator):
    symbol = "log"
    def _forward(self):
        return self.operands[0].data.log()
    def _backward(self):
        return (1.0 / self.operands[0].data) if self._op_requires_grad[0] else None