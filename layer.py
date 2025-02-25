from typing import Collection

from neuron import Neuron


class Layer:
    def __init__(self, inputs: int, outputs: int):
        self.neurons = [Neuron(inputs) for _ in range(outputs)]

    def __call__(self, x: Collection):
        ret = [n(x) for n in self.neurons]
        return ret[0] if len(ret) == 1 else ret

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

