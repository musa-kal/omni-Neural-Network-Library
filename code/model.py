"""
Author: Musa Kaleem
"""

import numpy as np

NP_FLOAT_PRECISION = np.float32

class ActivationFunctions:
    """
    Activation Functions class will hold different pre built activation function.
    """
    pass

# class Neuron:
#     def __init__(self, bias=0):
#         self.bias = bias

#     def feedforward(self, z: float) -> float:
#         return z + self.bias

class Layers:
    """
    Layers class holds logic for joining and infracting with joint layers and some other helper classes.
    """

    class BaseLayer:
        """
        Base Layer class used to create different types of hidden layers
        """
        def __init__(self):
            self.shape = ()
            self.name = None

        def feedforward(self):
            """
            should return a numpy array
            """
            raise NotImplementedError(f"feedforward must be implemented in the child class {self.name}!")
        
        def feedbackwards(self):
            """
            still figuring out how the returns should work
            """
            raise NotImplementedError(f"feedbackwards must be implemented in the child class {self.name}!")
        
        def __repr__(self):
            """
            return string including the layer name and shape
            """
            return f"{self.name}-{self.shape}"


    class DenseLayer(BaseLayer):
        """
        Dense Layer class inherits from BaseLayer and represents a dense layer
        """
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
            """
            feeds the input data forward through layer and returns the output a numpy array
            """
            if input_array.shape[0] != self.weights.shape[1]:
                raise ValueError(f"Input array shape {input_array.shape} doesn't match the shape of the layers weights shape {self.weights.shape}")
            
            self.prev_input = input_array
            
            z = np.dot(self.weights, input_array) + self.neurons
            
            if not np.all(np.isfinite(z)):
                raise ValueError("NaN or Inf detected in forward pass")

            if save:
                self.pre_activation = z.copy()

            if self.activation_function:
                z = self.activation_function(z)

            if save:
                self.post_activation = z.copy()

            return z
        

        def feedbackwards(self, lose_derivatives):
            """
            accepts a lose_derivatives which is ∂L/∂a derivative of lose function with respect to the 
            activation of current layer and returns ∂L/∂a(L-1), ∂L/∂w, ∂L/∂b
            """
            if lose_derivatives.shape != self.shape:
                raise ValueError(f"lose_derivatives {lose_derivatives.shape} doesn't equal current layer shape {self.shape}")
            

            if self.activation_function:
                activation_function_derivatives = self.activation_function.derivative(self.post_activation)
            else:
                activation_function_derivatives = np.ones(shape=self.shape, dtype=NP_FLOAT_PRECISION)
            
            dL_dz = activation_function_derivatives * lose_derivatives

            dz_dw = np.tile(self.prev_input, (self.shape[0],1))

            # ∂L/∂a(L-1), ∂L/∂w, ∂L/∂b
            return np.dot(self.weights.T, dL_dz), dz_dw.T * dL_dz, dL_dz

            
        def clear_all_saved_data(self):
            """
            helper function to clear all saved data within the current layer
            """
            self.pre_activation = None
            self.post_activation = None
            self.prev_input = None


    def __init__(self, input_shape:tuple):
        self.input_shape = input_shape
        self.layers = []        

    def join_front(self, new_layer: BaseLayer):
        """
        takes a BaseLayer class and joins it to the front/right of the network.
        """

        if len(self.layers) == 0:
            if len(new_layer.shape) != len(self.input_shape):
                raise ValueError(f"new layer shape {new_layer.shape} doesn't match the input shape {self.input_shape}!")
            self.layers.append(new_layer)
            return
        
        if len(new_layer.shape) != len(self.layers[-1].shape):
            raise ValueError(f"new layer shape {new_layer.shape} doesn't match the previous layer {self.layers[-1].shape}!")
        self.layers.append(new_layer)


    def feedforward(self, input_data):
        """
        takes input data and feeds it through the network returning the output.
        """
        if input_data.shape != self.input_shape:
            raise ValueError(f"input_data shape {input_data.shape} doesn't match the shape specified f{self.input_shape}")
        

        next_input = input_data
        for layer in self.layers:
            curr_output = layer.feedforward(next_input)
            next_input = curr_output
        
        return next_input

    def propagate_backwards(self, layer_output):
        
        layer_derivate = []
        next_input = layer_output

        for layer in self.layers[::-1]:
            output = layer.feedbackwards(next_input)
            layer_derivate.append(output)
            next_input = output[0]

        return layer_derivate[::-1]

    def __repr__(self):
        """
        string representing the network.
        """
        output = f"layers: [Input-{self.input_shape}\t" 
        for layer in self.layers:
            output += f"{layer}\t"
        return output.strip() + "]"
    
    

if __name__ == '__main__':
    # x = np.array([
    #     [1,3],
    #     [2,1],
    #     [1,3]
    # ])
    # y = np.array(
    #     [1,2,3]
    # )
    # print(np.dot(x.T,y))
    x = Layers(input_shape=(1,))
    l = Layers.DenseLayer(3)
    l.weights=np.array([[1],[0],[-1]])
    x.join_front(l)
    alp = 0.001
    
    output = x.feedforward(np.array([2.3514], dtype=NP_FLOAT_PRECISION))
    print(output)
    output = output - np.max(output)

    output = np.exp(output, dtype=NP_FLOAT_PRECISION)
    _sum = np.sum(output)
    _y = output / _sum
    y = np.array([0,1,0], dtype=NP_FLOAT_PRECISION)
    print(_y)
    loss = _y-y 
    print(loss)
    print("===")
    out = x.propagate_backwards(loss)
    for e in out[0]:
        print(e)
    
    pass
