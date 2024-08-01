"""
heres a function that can train the model
"""


from chrisNet.tensor import Tensor
from chrisNet.nn import NeuralNet
from chrisNet.loss import Loss, MSELoss
from chrisNet.optim import Optimizer, SGD
from chrisNet.data import DataIterator, BatchIterator

def train(net: NeuralNet,
        inputs: Tensor,
        target: Tensor,
        numEpochs: int = 5000, # fixed number of training times (passes)
        iterator: DataIterator = BatchIterator(),
        loss: Loss = MSELoss(),
        optimizer: Optimizer = SGD()) -> None:
    
    for epoch in range(numEpochs): #iterates for all epochs in the range of numEpochs (ex. 5000)
        epochLoss = 0.0 #First set all epochs (epoch is one passthrough the neural network)
        for batch in iterator(inputs, target): #for the batch
            predicted = net.forward(batch.inputs) #set predicted to the output of the forward func passing in the input batch
            epochLoss += loss.loss(predicted, batch.targets)    
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad) #push gradient backward 
            optimizer.step(net) #optimizes NeuralNet's params every passthrough 
        print(epoch, epochLoss) #prints the epoch and the loss from the passthrough