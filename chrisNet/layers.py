"""
Out neural nets are made up of 
Each layer needs to pass inputs forward and propgate gradients backward

Ex. 
input -> Linear -> Tanh -> Linear -> output :)
Note:
self allows a method to access data in the class (like @Published)
"""


from typing import Dict, Callable
import numpy as np
from chrisNet.tensor import Tensor

#Base class therefore "raise NotImplementedError" like a guard in Swift
class Layer:
    #constructor is necessary for some reason 
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        
    def forward(self, inputs: Tensor) -> Tensor:
        """
        produce output corresponding to inputs
        """
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        back propagate this gradient through 
        """
        raise NotImplementedError
#first layer
class Linear(Layer):
    """
    computes output = inputs * weight + bias
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        #inputs will be (batch_size, input_size)
        #outputs will be (batch_size, output_size)
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        @: matrix multiplcation
        outputs = inputs @ w + b
        """
        #save a copy
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"] #input times weight plus bias 
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) = f'(x) * b.T (transpose)
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """

        self.grads["b"] = np.sum(grad, axis = 0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T
    
F = Callable[[Tensor], Tensor]
    
class Activation(Layer):
    """
    An activation layer applies a function elementwise to its inputs
    F type takes a tensor and returns a tensor
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def tanh(self, x: Tensor) -> Tensor:
        return np.tanh(x)
    
    def tanh_prime(self, x: Tensor) -> Tensor:
        y = tanh(x)
        return 1 - y ** 2
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad

class tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)