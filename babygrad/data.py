from abc import ABC, abstractmethod

class Data(ABC):
    @abstractmethod
    def __init__(self, data):
        pass
    @abstractmethod
    def item(self):
        """Checks if self contains a single element and returns it. If self has multiple elements it raises a ValueError"""
        pass
    # class requires Add, Mul, Divide, Pow
    @abstractmethod
    def __add__(self, other):
        pass
    @abstractmethod
    def __radd__(self, other):
        pass
    @abstractmethod
    def __sub__(self, other):
        pass
    @abstractmethod
    def __rsub__(self, other):
        pass
    @abstractmethod
    def __mul__(self, other):
        pass
    @abstractmethod
    def __rmul__(self, other):
        pass
    @abstractmethod
    def __truediv__(self, other):
        pass
    @abstractmethod
    def __rtruediv__(self, other):
        pass
    @abstractmethod
    def __pow__(self, other):
        pass
    @abstractmethod
    def __rpow__(self, other):
        pass
    @abstractmethod
    def __neg__(self):
        pass
    @abstractmethod
    def __gt__(self, other):
        pass
    
    @abstractmethod
    def zero(self):
        """Returns a 'zero' element of the same type and dimension as self."""
    
    @abstractmethod
    def one(self):
        """Returns a 'one' element of the same type and dimension as self."""