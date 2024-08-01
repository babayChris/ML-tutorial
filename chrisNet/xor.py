"""
xor is a classic example of a function that cannot be learned with a simple linear model
"""

import numpy as np

from chrisNet.test import train
from chrisNet.nn import NeuralNet
from chrisNet.layers import Linear, Tanh

inputs = np.array([# inputs into network
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

targets = np.array([ #desired outcomes, left is false, right is true
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])

net = NeuralNet([
    Linear(input_size = 2, output_size = 2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)




