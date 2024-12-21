import math


class Value:
    def __init__(self, data, _children=[], _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = list(_children)
        self._op = _op
        self._backward = None

    def __repr__(self):
        return f'Value(data={self.data:.6f})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ret = Value(self.data + other.data, [self, other], '+')
        def _backward():
            self.grad += ret.grad
            other.grad += ret.grad
        ret._backward = _backward
        return ret

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ret = Value(self.data * other.data, [self, other], '*')
        def _backward():
            self.grad += ret.grad * other.data
            other.grad += ret.grad * self.data
        ret._backward = _backward
        return ret

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __pow__(self, power, modulo=None):
        ret = Value(self.data ** power, [self], "pow")
        def _backward():
            self.grad = power * (self.data ** (power - 1)) * ret.grad
        ret._backward = _backward
        return ret

    def __truediv__(self, other):
        return self * (other ** -1)

    def exp(self):
        x = self.data
        ret = Value(math.exp(x), [self], 'exp')
        def _backward():
            self.grad += ret.data * ret.grad
        ret._backward = _backward
        return ret

    def tanh(self):
        ret = Value((math.exp(self.data) - math.exp(-self.data)) / (math.exp(self.data) + math.exp(-self.data)), [self], "tanh")
        def _backward():
            self.grad = (1 - ret.data ** 2) * ret.grad
        ret._backward = _backward
        return ret
    def backward(self):
        self.grad = 1.0
        topo = []
        visited = {self}
        def visit(v):
            for i in v._prev:
                if i not in visited:
                    visited.add(i)
                    visit(i)
            topo.append(v)
        visit(self)
        print(f'Backpropagation: {len(topo)} nodes')
        for i in reversed(topo):
            if not i._backward is None:
                i._backward()


if __name__ == '__main__':

    a = Value(3.0)
    b = Value(4.0)
    d = Value(5.0)
    c = a * b
    e = c + d
    e.backward()

    x = Value(5)
    y = x / 3
    y.backward()

    print(y)
