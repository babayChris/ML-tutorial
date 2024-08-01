"""
We use an optimzer to adjust the parameters of out network based on gradients computed during backpropagation
"""

from chrisNet.nn import NeuralNet

class Optimizer:
    def step(self, netL: NeuralNet) -> None:
        raise NotImplementedError
    

class SGD(Optimizer):
    def __init__(self, learningRate: float = 0.01) -> None:
        self.learningRate = learningRate

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.learningRate * grad
            """
            pushes parameter in the opposite of the direction where the function grows the fastest
            learningRate is set to 0.01 to take small steps to not overstep, multiplied by the grad to get the direction 
            and subtracting to push in the opposite direction
            """
