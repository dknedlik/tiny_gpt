from layer import Layer
from value import Value
from draw import draw_dot


class MLP:
    def __init__(self, inputs, outputs):
        sz = [inputs] + outputs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


n = MLP(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs


for k in range(1000):
    # forward pass
    y_pred = [n(x) for x in xs]
    loss: Value
    loss = sum([(y_out - ygt)**2 for ygt, y_out in zip(ys, y_pred)])
    # backward pass
    for p in n.parameters():
        p.gradient = 0.0
    loss.backward()
    # update
    for p in n.parameters():
        p.internal += -.05 * p.gradient


y_pred = [n(x) for x in xs]
loss = sum([(y_out - ygt) ** 2 for ygt, y_out in zip(ys, y_pred)])
print(loss,y_pred)
# dot = draw_dot(loss)
# dot.render(filename='dot.png', view=True)
