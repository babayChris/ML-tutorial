"""
we feed inputs into our network in batches
so here are tools for iterating over data in batches
"""

import numpy as np
from typing import Iterator, NamedTuple
from chrisNet.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError
    
class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None: #shuffle is true to shuffle batches after every layer
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]: #when called in another class this is what happens
        start = np.arrange(0, len(inputs), self.batch_size) #0 - start, len(inputs) - end, self.batch_size - step size
        if self.shuffle: #if self.shffle bool is true
            np.random.shuffle(start)#shuffles start which is the numpy arrange function

        for start in start:
            end = start + self.batch_size
            batch_inputs = inputs[start,end] #sets batch inputs to inputs Tensor iterating from start to end
            batch_targets = targets[start,end]#sets batch targets to targets Tensor iterating from start to end
            yield Batch(batch_inputs, batch_targets)
            
