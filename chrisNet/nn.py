"""
A NeuralNet is just a collection of layers
It behaves a lot like a layer itself, although
we're not going to make it one
"""

from typing import Sequence, Iterator, Tuple
from chrisNet.tensor import Tensor
from chrisNet.layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    #pushes inputs through one layer at a time
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs 
    
    #pushes grad through backward layers until it reaches the front of the network
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layers.param.items():
                grad = layer.grads[name]
                yield param, grad
