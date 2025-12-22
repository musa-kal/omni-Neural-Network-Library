import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from .layers import BaseLayer

class Sequential:
    """
    Sequential class holds logic for joining and infracting with joint layers and some other helper classes.
    Essentially acts as a container or 'Sequential' model wrapper for individual layers.
    """

    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape: Tuple[int, ...] = input_shape
        self.layers: List[BaseLayer] = []        

    def join_front(self, new_layer: BaseLayer) -> None:
        """
        takes a BaseLayer class and joins it to the front/right of the network.
        Essentially 'append' a layer to the stack.
        """

        if len(self.layers) == 0:
            # First layer must match the network's input shape
            if len(new_layer.shape) != len(self.input_shape):
                raise ValueError(f"new layer shape {new_layer.shape} doesn't match the input shape {self.input_shape}!")
            new_layer.init_params(self.input_shape)
            self.layers.append(new_layer)
            return
        
        # Subsequent layers must match the output shape of the previous layer
        if len(new_layer.shape) != len(self.layers[-1].shape):
            raise ValueError(f"new layer shape {new_layer.shape} doesn't match the previous layer {self.layers[-1].shape}!")
        new_layer.init_params(self.layers[-1].shape)
        self.layers.append(new_layer)


    def feedforward(self, input_data: npt.NDArray[np.float32], save: bool = False) -> npt.NDArray[np.float32]:
        """
        takes input data and feeds it through the network returning the output.
        Passes data sequentially through all layers.
        """
        if input_data.shape != self.input_shape:
            raise ValueError(f"input_data shape {input_data.shape} doesn't match the shape specified f{self.input_shape}")
        

        next_input = input_data
        for layer in self.layers:
            curr_output = layer.feedforward(next_input, save)
            next_input = curr_output
        
        return next_input

    def propagate_backwards(self, layer_output: npt.NDArray[np.float32]) -> List[List[npt.NDArray[np.float32]]]:
        """
        Backpropagates the error signal from the last layer to the first.
        Returns a list of gradients for every layer.
        """
        layer_derivate: List[List[npt.NDArray[np.float32]]] = []
        next_input = layer_output

        # Iterate layers in reverse order
        for layer in self.layers[::-1]:
            output = layer.feedbackwards(next_input)
            # Store [Weight Gradients, Bias Gradients]
            layer_derivate.append([output[1], output[2]])
            # Pass the error for the previous layer (dL/dx) to the next iteration
            next_input = output[0]

        # Reverse list back to forward order so it matches self.layers indices
        return layer_derivate[::-1]
    
    def adjust_layer_parameter(self, i: int, gradients: List[npt.NDArray[np.float32]]) -> None:
        if i < 0 or i >= len(self.layers):
            raise ValueError(f"i must be with in range [0, {len(self.layers)})")
        self.layers[i].adjust_parameters(gradients) 

    def __repr__(self) -> str:
        """
        string representing the network.
        """
        output = f"layers: [Input-{self.input_shape}\t" 
        for layer in self.layers:
            output += f"{layer}\t"
        return output.strip() + "]"