"""
loss functions check how accurate predictions are 
can use to adjust parameters of out network
"""
import numpy as np
#Imports Tensor clas s from tensor.py
from chrisNet.tensor import Tensor

#allows for future implementation of other loss functions
class Loss:
    #Creates method that takes in a prediction as a Tensor and an actual value as a Tensor
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
"""
Here we use Total Square Error instead of Means Square Error
"""
class MSELoss(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2) #np.sum - sums array elements over a given axis
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual) #Derivative of loss function and x = predicted - actual



