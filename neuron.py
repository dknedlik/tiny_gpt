import random
from typing import Union, Collection

from value import Value


class Neuron:
    def __init__(self, inputs: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: Union[Collection]) -> Value:
        # w * x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh()

    def parameters(self) -> Collection:
        return self.w + [self.b]
