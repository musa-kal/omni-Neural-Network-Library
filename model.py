import numpy as np

NP_FLOAT_PRECISION = np.float32

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
        
        def feedbackwards(self):
            raise NotImplementedError("feedbackwards must be implemented in the child class!")
        
        def __repr__(self):
            return f"{self.name}-{self.shape}"

    class DenseLayer(BaseLayer):
        def __init__(self, neuron_count:int=1, activation_function:ActivationFunctions|None=None, input_shape:tuple=(1,)):
            if neuron_count < 1:
                raise ValueError("neuron_count must be greater then 0!")
            self.shape = (neuron_count,)
            self.neurons = np.empty(shape=(neuron_count), dtype=NP_FLOAT_PRECISION) # neuron biases stored as np array
            self.name = "Dense Layer"
            self.activation_function = activation_function
            self.weights = np.empty(shape=(neuron_count, input_shape[0]), dtype=NP_FLOAT_PRECISION) # weights represented as 2nd numpy array rows representing the current layer neuron index and column previous inputs
            self.pre_activation = None
            self.post_activation = None
            self.prev_input = None
        
        def feedforward(self, input_array, save=False):
            if input_array.shape[0] != self.weights.shape[1]:
                raise ValueError(f"Input array shape {input_array.shape} doesn't match the shape of the layers weights shape {self.weights.shape}")
            
            self.prev_input = input_array
            
            z = np.dot(self.weights, input_array) + self.neurons

            if save:
                self.pre_activation = z.copy()

            if self.activation_function:
                z = self.activation_function(z)

            if save:
                self.post_activation = z.copy()

            return z
        
        def feedbackwards(self, lose_derivatives):
            if lose_derivatives.shape != self.shape:
                raise ValueError(f"lose_derivatives {lose_derivatives.shape} doesn't equal current layer shape {self.shape}")
            
            activation_function_derivatives = np.ones(shape=self.shape, dtype=NP_FLOAT_PRECISION)

            if self.activation_function:
                activation_function_derivatives = self.activation_function.derivative(self.post_activation)

            
            

            
        def clear_all_saves(self):
            self.pre_activation = None
            self.post_activation = None
            self.prev_input = None


    def __init__(self, input_shape:tuple):
        self.input_shape = input_shape
        self.layers = []        

    def join_front(self, new_layer: BaseLayer):

        if len(self.layers) == 0:
            if len(new_layer.shape) != len(self.input_shape):
                raise ValueError(f"new layer shape {new_layer.shape} doesn't match the input shape {self.input_shape}!")
            self.layers.append(new_layer)
            return
        
        if len(new_layer.shape) != len(self.layers[-1].shape):
            raise ValueError(f"new layer shape {new_layer.shape} doesn't match the previous layer {self.layers[-1].shape}!")
        self.layers.append(new_layer)

    def feedforward(self, input_data):
        if input_data.shape != self.input_shape:
            raise ValueError(f"input_data shape {input_data.shape} doesn't match the shape specified f{self.input_shape}")
        
        layer_outputs = []

        next_input = input_data
        for i, layer in enumerate(self.layers):
            curr_output = layer.feedforward(next_input)
            layer_outputs.append(curr_output)
            next_input = curr_output
        
        return layer_outputs

    def propagate_backwards(self, layer_outputs):
        pass

    def __repr__(self):
        output = f"layers: [Input-{self.input_shape}\t" 
        for layer in self.layers:
            output += f"{layer}\t"
        return output.strip() + "]"
    
    
    


if __name__ == '__main__':
    input = Layers.DenseLayer(1)
    x = Layers(input_shape=(2,))
    l = Layers.DenseLayer(3)
    l.weights=np.array([[1,1],[2,1],[1,-1]])
    x.join_front(l)
    print(x.feedforward(np.array([1,2])))
    pass