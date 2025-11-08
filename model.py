import numpy as np

class ActivationFunctions:
    pass

class Neuron:
    def __init__(self, bias=0):
        self.bias = bias
        self.activation_function = None
        self.weights = np.array([1])

    def feedforward(self, z: float) -> float:
        output = np.sum(np.dot(self.weights, z)) + self.bias
        return output

class BaseLayer:
    def __init__(self):
        self.shape = ()

class DenseLayer(BaseLayer):
    def __init__(self, neuron_count:int=1):
        if neuron_count < 1:
            raise ValueError("neuron_count must be greater then 0!")
        self.shape = tuple([neuron_count])
        self.neurons = [Neuron() for _ in range(neuron_count)]


class Layers:
    def __init__(self, first_layer: BaseLayer, activation_function: str=None):
        self.layers = [first_layer]
        self.weights = []
        self.activation_function = activation_function if activation_function else None
    
    def join_front(self, new_layer: BaseLayer):
        if len(new_layer.shape) == len(self.layers[-1].shape):
            raise ValueError("new layer shape doesn't match the previous layer!")
        self.layers.append(new_layer)
        self.weights.append(np.empty(shape=(new_layer.shape[0], self.layers[-1].shape[0])))


    
    
    
    


if __name__ == '__main__':
    pass