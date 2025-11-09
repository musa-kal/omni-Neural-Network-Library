import numpy as np

NP_FLOAT_PRECISION = np.float64

class ActivationFunctions:
    pass

# class Neuron:
#     def __init__(self, bias=0):
#         self.bias = bias

#     def feedforward(self, z: float) -> float:
#         return z + self.bias

class Layers:
    class BaseLayer:
        """
        Base Layer Class used to create different types of hidden layers
        """
        def __init__(self):
            self.shape = ()
            self.name = None

        def feedforward(self):
            raise NotImplementedError("feedforward must be implemented in the child class!")
        
        def __repr__(self):
            return f"{self.name}-{self.shape}"

    class DenseLayer(BaseLayer):
        def __init__(self, neuron_count:int=1, activation_function:str|None=None):
            if neuron_count < 1:
                raise ValueError("neuron_count must be greater then 0!")
            self.shape = tuple([neuron_count])
            self.neurons = np.empty(shape=(neuron_count), dtype=NP_FLOAT_PRECISION) # neuron biases stored as np array
            self.name = "Dense Layer"
            self.activation_function = activation_function
        
        def feedforward(self, input_array):
            z = input_array + self.neurons
            if self.activation_function:
                z = self.activation_function(z)
            return z

    def __init__(self, input_shape:tuple):
        self.input_shape = tuple(input_shape)
        self.layers = []
        self.weights = []

    def __init_first_layer(self, layer:BaseLayer):
        if len(layer.shape) != len(self.input_shape):
            raise ValueError(f"new layer shape {layer.shape} doesn't match the input shape {self.input_shape}!")
        self.layers.append(layer)
        self.weights.append(np.empty(shape=(self.input_shape[0], layer.shape[0]), dtype=NP_FLOAT_PRECISION))



    
    def join_front(self, new_layer: BaseLayer):
        if len(self.layers) == 0:
            self.__init_first_layer(new_layer)
            return
        if len(new_layer.shape) != len(self.layers[-1].shape):
            raise ValueError(f"new layer shape {new_layer.shape} doesn't match the previous layer {self.layers[-1].shape}!")
        self.layers.append(new_layer)
        self.weights.append(np.empty(shape=(new_layer.shape[0], self.layers[-1].shape[0]), dtype=NP_FLOAT_PRECISION))

    def feedforward(self, input_data):
        if input_data.shape != self.input_shape:
            raise ValueError(f"input_data shape {input_data.shape} doesn't match the shape specified f{self.input_shape}")
        
        layer_outputs = []

        curr_input = input_data
        for i, layer in enumerate(self.layers):
            pass 

    def __repr__(self):
        output = f"layers: [Input-{self.input_shape}\t" 
        for layer in self.layers:
            output += f"{layer}\t"
        return output.strip() + "]"
    
    
    


if __name__ == '__main__':
    input = Layers.DenseLayer(1)
    print(input.feedforward([1]))
    x = Layers(input_shape=[1])
    x.join_front(Layers.DenseLayer(3))
    print(x)
    pass