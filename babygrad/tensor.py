import math
import numpy as np
from babygrad import log
from abc import ABC, abstractmethod
#from babygrad.data.float import FloatData as Data
from babygrad.data.numpy import NumpyData as Data
class Operand(ABC):
    symbol: str
    data: Data
    grad: Data
    def __init__(self, requires_grad = False):
        self.requires_grad = requires_grad
        self._decendant_requires_grad = None
        self.grad = None
    def item(self) -> float:
        """Returns a single element from the data as a python float."""
        return self.data.item()
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data):
        """Sets the data of this node to the given data."""
        self._data = data if isinstance(data, Data) else Data(data)
    @property
    def grad(self):
        return self._grad
    @grad.setter
    def grad(self, grad):
        """Sets the gradient of this node to the given gradient."""
        self._grad = grad if isinstance(grad, Data) or grad is None else Data(grad)
    def zero_grad(self):
        if self.grad is not None:
            self.grad = self.grad.zero()
    def zero_grads(self):
        """Sets the gradient of this node and all its descendants to None."""
    
    def __repr__(self):
        return f"{type(self).__name__}({self.data})"
    
    @abstractmethod
    def __call__(self):
        """Computes the whole computation graph and assigns self.data for each node."""
        pass
    
    def _binary_op(self, operand, Op: 'Operator', swap=False):
        match operand:
            case Operand():
                pass
            case _:
                operand = Tensor(data=operand)
        #self._check_binary_op(operand)
        return Op([operand, self]) if swap else Op([self, operand])
    
    def _unary_op(self, Op):
        return Op([self])
    
    # ****** Atomic operations ********
    def __add__(self, addend):
        return self._binary_op(addend, Add)
    __radd__ = __add__
    def __mul__(self, multiplicant: 'Tensor'):
        return self._binary_op(multiplicant, Mul)
    __rmul__ = __mul__
    def __truediv__(self, divisor: 'Tensor'):
        return self._binary_op(divisor, Div)
    def __rtruediv__(self, dividend: 'Tensor'):
        return self._binary_op(dividend, Div, swap=True)
    def __pow__(self, exponent: 'Tensor'):
        return self._binary_op(exponent, Pow)
    def __rpow__(self, base: 'Tensor'):
        return self._binary_op(base, Pow, swap=True)
    def __sub__(self, subtrahend: 'Tensor'):
        return self._binary_op(subtrahend, Sub)
    def __rsub__(self, minuend: 'Tensor'):
        return self._binary_op(minuend, Sub, swap=True)
    def __neg__(self):
        return Neg([self])
    
    def maximum(self, other):
        return self._binary_op(other, Maximum)
    
    def exp(self):
        return self._unary_op(Exp)
    def log(self):
        return self._unary_op(Log)

    def permute(self, *dims: int):
        self._check_permute(dims)
        return Permute([self], dims=dims)
    def reshape(self, *shape: int):
        self._check_reshape(shape)
        if shape == self.shape:
            return self
        return Reshape([self], shape=shape)
    
    def expand(self, *shape):
        self._check_expand(shape)
        return Expand([self], shape=shape)
    def sum(self, *dims: int):
        dims = tuple(sorted(map(lambda x: x if x >= 0 else len(self.shape) + x, dims)))
        self._check_sum(dims)
        if len(dims) == 0:
            return self
        return Sum([self], dims=dims)
    
    
    # ****** Non atomic operations *******
    # Those are operations that can be composed of atomic operations
    # TODO: They can be overriden, by specific Data implementations, to be more efficient
    def sqrt(self):
        return self**0.5
    def sigmoid(self):
        return (1.0 / (1.0 + (-self).exp()))
    def relu(self):
        return self._binary_op(0, Maximum)
    def tanh(self):
        return 2.0 * ((2.0 * self).sigmoid()) - 1.0
    def __matmul__(self, other):
        j = other.shape[1]
        k = self.shape[0]
        n = self.shape[1]
        return (self.reshape(k, 1, n).expand(k, j, n) * other.permute(1,0).reshape(1, j, n).expand(k, j, n)).sum(2).reshape(k, j)
    
    # ****** Input checks ********
    def _check_permute(self, dims):
        if len(dims) != len(self.shape):
            raise ValueError(f"Permutation {dims} doesn't match length of {self.shape}.")
        seen = set()
        for dim in dims:
            if dim < 0 or dim >= len(self.shape):
                raise ValueError(f"Dimension [{dim}] is not valid for shape {self.shape}.")
            if dim in seen:
                raise ValueError(f"Dimension [{dim}] is duplicate in {dims}.")
    def _check_reshape(self, shape):
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError(f"Shape {shape} is not valid for shape {self.shape}.")
    
    def _check_expand(self, shape):
        if len(shape) != len(self.shape):
            raise ValueError(f"Shape {shape} does not have the same length as {self.shape}.")
        for i in range(len(shape)):
            if self.shape[i] != 1 and self.shape[i] != shape[i]:
                raise ValueError(f"Shape {shape} doesn't match {self.shape} at dim [{i}].")
    def _check_sum(self, dims):
        seen = set()
        for dim in dims:
            if dim < 0 or dim >= len(self.shape):
                raise ValueError(f"Dimension [{dim}] is not valid for shape {self.shape}.")
            if dim in seen:
                raise ValueError(f"Dimension [{dim}] is duplicate in {dims}.")
    def _check_binary_op(self, operand):
        if not isinstance(operand, Operand):
            raise ValueError(f"Operand must be of type Operand, not {type(operand)}.")
        # TODO: implement broadcasting
        if len(self.shape) != len(operand.shape):
            raise ValueError(f"Shapes {self.shape} and {operand.shape} don't match.")
        for i in range(len(self.shape)):
            if self.shape[i] != operand.shape[i]:
                raise ValueError(f"Shapes {self.shape} and {operand.shape} don't match at dim [{i}].")

# ************ Lazy tensors ************
class LazyTensor(Operand):
    symbol="Lazy"
    def __init__(self, requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self._data = None
        self.grad = None
    def __call__(self):
        return self.data
    def zero_grads(self):
        self.zero_grad()
    @property
    def data(self):
        if self._data is None:
            self.data = self._create()
        return self._data
    @data.setter
    def data(self, value):
        self._data = value if isinstance(value, Data) else Data(value)
    @abstractmethod
    def _create(self) -> Data:
        pass
class Tensor(LazyTensor):
    symbol = "T"
    def __init__(self, data: Data | list, requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        if not hasattr(data, "shape"):
            self.shape = np.array(data).shape
        else:
            self.shape = data.shape
        self.__data = data
    def _create(self):
        return Data(self.__data)
    @staticmethod
    def zeros(shape):
        return Zeros(shape=shape)
    @staticmethod
    def ones(shape):
        return Ones(shape)
    @staticmethod
    def zeros_like(value):
        return Tensor.zeros(value.shape)
    @staticmethod
    def ones_like(value):
        return Tensor.ones(value.shape)
    @staticmethod
    def uniform(low, high, shape):
        return Uniform(low, high, shape)
    @staticmethod
    def normal(mean, std, shape):
        return Normal(mean, std, shape)
class Zeros(LazyTensor):
    symbol = "0s"
    def __init__(self, shape: tuple[int, ...], requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self.shape = shape
    def _create(self):
        return Data.zeros(self.shape)
class Ones(LazyTensor):
    symbol = "1s"
    def __init__(self, shape: tuple[int, ...], requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self.shape = shape
    def _create(self):
        return Data.ones(self.shape)
class Uniform(LazyTensor):
    symbol = "Uniform"
    def __init__(self, low: float, high: float, shape: tuple[int, ...], requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self.low, self.high, self.shape = low, high, shape
    def _create(self):
        return Data.uniform(self.shape) * (self.high - self.low) + self.low
class Normal(LazyTensor):
    symbol = "Normal"
    def __init__(self, mean: float, std: float, shape: tuple[int, ...], requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self.mean, self.std, self.shape = mean, std, shape
    def _create(self):
        return Data.normal(self.shape) * self.std + self.mean


class Operator(Operand):
    symbol = "O"
    def __init__(self, operands: list[Operand], requires_grad = False):
        super().__init__(requires_grad=requires_grad)
        self._topo_sorted_graph = None
        self.operands = operands
        self.data = self._forward()
        self.grad = None
    
    @classmethod
    def _dfs(cls, current: Operand, visited: set[Operand], seq: list['Operator']):
        if current in visited:
            return
        visited.add(current)
        if isinstance(current, Operator):
            for op in current.operands:
                cls._dfs(op, visited, seq)
            seq.append(current)
    
    def __call__(self, compute_data=True):
        """Forward pass of the computation graph."""
        log.debug(f"Forward pass of {self} ({compute_data=})")
        # create a topo sort of the computation graph
        self._topo_sorted_graph: list[Operator] = []
        self._dfs(self, set(), self._topo_sorted_graph)
        
        for operator in self._topo_sorted_graph:
            log.debug(f"Forwarding {operator}")
            operator._op_requires_grad = []
            for operand in operator.operands:
                log.debug(f"Operand {operand}({operand.requires_grad=}, {operand._decendant_requires_grad=})")
                oprg = operand.requires_grad or operand._decendant_requires_grad
                operator._op_requires_grad.append(oprg)
                if oprg and operand.grad is None:
                    # create the gradient if it does not exist yet
                    operand.grad = operand.data.zero()
            operator._decendant_requires_grad = any(operator._op_requires_grad)
            log.debug(f"Operands of {operator} require grad: {operator._op_requires_grad}. Therefore {operator} requires grad: {operator._decendant_requires_grad}.")
            if compute_data:
                operator.data = operator._forward()
    
    @abstractmethod
    def _forward(self) -> Data:
        """Computes only this node of the computation graph."""
    
    def backward(self):
        if self._topo_sorted_graph is None:
            self.__call__(compute_data=False)
        self.grad = self.data.ones_like(self.data)
        # Compute gradients in reverse topo sort order
        for operator in reversed(self._topo_sorted_graph):
            log.debug(f"Backward pass of {operator} ({operator.grad=})")
            operand_gradients = operator._backward()
            # Add the weighted gradients to the child nodes
            for operand, operand_gradient in zip(operator.operands, operand_gradients):
                if operand_gradient is None:
                    continue
                operand.grad += operand_gradient
    
    @abstractmethod
    def _backward(self) -> list[Data | None]:
        """Computes the gradients of this nodes operands if they are required."""
    
    def zero_grads(self):
        self.zero_grad()
        for operand in self.operands:
            operand.zero_grads()
        

# ********* Atomic binary operations *************

class ShapePreservingOperator(Operator):
    def __init__(self, operands: list[Operand], requires_grad = False):
        super().__init__(operands, requires_grad=requires_grad)
        self.shape = operands[0].shape

class Add(ShapePreservingOperator):
    symbol = "+"
    def _forward(self):
        return sum(op.data for op in self.operands)
    def _backward(self):
        return  self.operands[0].data.one() * self.grad if self._op_requires_grad[0] else None, \
                self.operands[1].data.one() * self.grad if self._op_requires_grad[1] else None
class Sub(ShapePreservingOperator):
    symbol = "-"
    def _forward(self):
        return self.operands[0].data - sum(op.data for op in self.operands[1:])
    def _backward(self):
        return  self.operands[0].data.one() * self.grad if self._op_requires_grad[0] else None, \
                -self.operands[1].data.one() * self.grad if self._op_requires_grad[1] else None
class Mul(ShapePreservingOperator):
    symbol = "*"
    def _forward(self):
        return self.operands[0].data * self.operands[1].data
    def _backward(self):
        return  self.operands[1].data * self.grad if self._op_requires_grad[0] else None, \
                self.operands[0].data * self.grad if self._op_requires_grad[1] else None
class Div(ShapePreservingOperator):
    symbol = "/"
    def _forward(self):
        return self.operands[0].data / self.operands[1].data
    def _backward(self):
        return  (1.0 / self.operands[1].data) * self.grad if self._op_requires_grad[0] else None, \
                -(self.operands[0].data / (self.operands[1].data ** 2)) * self.grad if self._op_requires_grad[1] else None
class Pow(ShapePreservingOperator):
    symbol = "**"
    def _forward(self):
        return self.operands[0].data ** self.operands[1].data
    def _backward(self):
        return  (self.operands[1].data * (self.operands[0].data ** (self.operands[1].data - 1))) * self.grad if self._op_requires_grad[0] else None, \
                ((self.operands[0].data ** self.operands[1].data) * self.operands[0].data.log()) * self.grad if self._op_requires_grad[1] else None
class Maximum(ShapePreservingOperator):
    symbol = "max"
    def _forward(self):
        return max(self.operands[0].data, self.operands[1].data)
    def _backward(self):
        max_op = max(self.operands, key=lambda op: op.data)
        return  self.operands[0].data.one() if max_op == self.operands[0] else self.operands[0].data.zero() * self.grad if self._op_requires_grad[0] else None, \
                self.operands[1].data.one() if max_op == self.operands[1] else self.operands[1].data.zero() * self.grad if self._op_requires_grad[1] else None


# ********* Atomic unary operations *************

class Neg(ShapePreservingOperator):
    symbol = "-"
    def _forward(self):
        return -self.operands[0].data
    def _backward(self):
        return -self.operands[0].data.one() * self.grad if self._op_requires_grad[0] else None,
class Exp(ShapePreservingOperator):
    symbol = "exp"
    def _forward(self):
        return self.operands[0].data.exp()
    def _backward(self):
        return self.data * self.grad if self._op_requires_grad[0] else None,
class Log(ShapePreservingOperator):
    symbol = "log"
    def _forward(self):
        return self.operands[0].data.log()
    def _backward(self):
        return (1.0 / self.operands[0].data) * self.grad if self._op_requires_grad[0] else None,


# ********* Atomic movement operations *************
class Permute(Operator):
    symbol = "permute"
    def __init__(self, operands: list[Operand], dims: tuple[int, ...], **kwargs):
        self.dims = dims
        self.shape = tuple(operands[0].shape[d] for d in dims)
        super().__init__(operands, **kwargs)
    def _forward(self):
        return self.operands[0].data.permute(self.dims)
    def _backward(self):
        # inverse of permute is permutation with argsort of dims as dims
        return self.grad.permute(sorted(sorted(self.dims), key=lambda i: self.dims[i])) if self._op_requires_grad[0] else None,

class Reshape(Operator):
    symbol = "reshape"
    def __init__(self, operands: list[Operand], shape: tuple[int, ...], **kwargs):
        self.shape = shape
        super().__init__(operands, **kwargs)
    def _forward(self):
        return self.operands[0].data.reshape(self.shape)
    def _backward(self):
        return self.grad.reshape(self.operands[0].shape) if self._op_requires_grad[0] else None,

# ********* Atomic reduction operations *************

class Expand(Operator):
    symbol = "expand"
    def __init__(self, operands: list[Operand], shape: tuple[int, ...], **kwargs):
        self.shape = shape
        super().__init__(operands, **kwargs)
    def _forward(self):
        return self.operands[0].data.expand(self.shape)
    def _backward(self):
        return self.grad.sum(dims=tuple(i for i, s in enumerate(self.operands[0].shape) if s != self.shape[i])) if self._op_requires_grad[0] else None,

class Sum(Operator):
    symbol = "sum"
    def __init__(self, operands: list[Operand], dims: tuple[int, ...], **kwargs):
        self.dims = dims
        self.shape = tuple(s if i not in dims else 1 for i, s in enumerate(operands[0].shape))
        super().__init__(operands, **kwargs)
    def _forward(self):
        return self.operands[0].data.sum(dims=self.dims)
    def _backward(self):
        return self.grad.expand(self.operands[0].shape) if self._op_requires_grad[0] else None,