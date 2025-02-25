import math
from typing import Union
from draw import draw_dot

class Value:
    def __init__(self, value: Union[int, float, str],
                 _children: Union[tuple['Value'], tuple['Value', 'Value']] = (),
                 _op: str = '',
                 label: str = ''):
        self.internal = value
        self.gradient = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other: Union['Value', int, float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        ret = Value(self.internal + other.internal, (self, other), '+')

        def _backward() -> None:
            self.gradient += 1.0 * ret.gradient
            other.gradient += 1.0 * ret.gradient

        ret._backward = _backward
        return ret

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: Union['Value', int, float]) -> 'Value':
        return self + (-other)

    def __mul__(self, other: Union['Value', int, float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        ret = Value(self.internal * other.internal, (self, other), '*')

        def _backward() -> None:
            self.gradient += other.internal * ret.gradient
            other.gradient += self.internal * ret.gradient

        ret._backward = _backward
        return ret

    def __rmul__(self, other: Union['Value', int, float]) -> 'Value':
        return self * other

    def __radd__(self, other: Union['Value', int, float]) -> 'Value':
        return self + other

    def __rsub__(self, other: Union['Value', int, float]) -> 'Value':
        return self - other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        ret = Value(self.internal ** other, (self,), f'**{other}')

        def _backward():
            self.gradient += other * (self.internal ** (other - 1)) * ret.gradient

        ret._backward = _backward

        return ret

    def __truediv__(self, other: Union['Value', int, float]) -> 'Value':
        return self * other ** -1

    def tanh(self) -> 'Value':
        x = self.internal
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        ret = Value(t, (self,), 'tanh')

        def _backward():
            self.gradient += (1 - t ** 2) * ret.gradient

        ret._backward = _backward

        return ret

    def exp(self) -> 'Value':
        x = self.internal
        ret = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.gradient += ret.internal * ret.gradient

        ret._backward = _backward
        return ret

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v: 'Value'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.gradient = 1.0
        node: Value
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f'Value(internal={self.internal}, gradient={self.gradient}, op={self._op}, label={self.label})'


# test code
# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# neuron bias
b = Value(6.8813735870195432, label='b')  # trust me bro
# x1*w1 + x2*w2 + b
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1.label = 'x1*w1'
x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b
n.label = 'n'

# o = n.tanh()
e = (n * 2).exp()
o = (e - 1) / (e + 1)
o.label = 'o'

o.backward()
dot = draw_dot(o)
# dot.render(filename='dot.png', view=True)
