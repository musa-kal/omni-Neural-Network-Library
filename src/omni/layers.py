import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional, Any
from .utils import NP_FLOAT_PRECISION, LayerSave

class BaseLayer:
    """
    Base Layer class used to create different types of hidden layers
    """
    
    @staticmethod
    def base_save_function(prev_save: Optional[LayerSave], 
                    prev_input: Optional[npt.NDArray[np.float32]], 
                    pre_activation: Optional[npt.NDArray[np.float32]], 
                    post_activation: Optional[npt.NDArray[np.float32]]) -> LayerSave:
            return LayerSave(prev_input, pre_activation, post_activation)
    
    def __init__(self) -> None:
        self.shape: Tuple[int, ...] = ()
        self.name: Optional[str] = None
        self.saved: Optional[LayerSave] = None # Holds the LayerSave object (cache) for this layer
        self.weights: Optional[npt.NDArray[np.float32]] = None
        self.biases: Optional[npt.NDArray[np.float32]] = None
            
        self.save_function = BaseLayer.base_save_function


    def feedforward(self, input_array: npt.NDArray[np.float32], save: bool = False) -> npt.NDArray[np.float32]:
        """
        should return a numpy array
        """
        raise NotImplementedError(f"feedforward must be implemented in the child class {self.name}!")
    
    def feedbackwards(self, lose_derivatives: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        should return a tuple containing ∂L/∂a(L-1), ∂L/∂w, ∂L/∂b
        """
        raise NotImplementedError(f"feedbackwards must be implemented in the child class {self.name}!")
    

    def clear_all_saved_data(self) -> None:
        """
        helper function to clear all saved data within the current layer
        Used to free memory after backpropagation is complete.
        """
        self.saved = None
    
    def adjust_parameters(self, gradients: List[npt.NDArray[np.float32]]) -> None:
        # Updates weights and biases using the calculated gradients
        raise NotImplementedError(f"adjust_parameters must be implemented in the child class {self.name}!")
    
    def init_params(self, input_shape: Tuple[int, ...]) -> None:
        # Initializes weights based on input size
        raise NotImplementedError(f"init_params must be implemented in the child class {self.name}!")


    def __repr__(self) -> str:
        """
        return string including the layer name and shape
        """
        return f"{self.name}-{self.shape}"


class DenseLayer(BaseLayer):
    """
    Dense Layer class inherits from BaseLayer and represents a dense layer
    (Fully Connected Layer) where every input connects to every neuron.
    """
    def __init__(self, 
                 neuron_count: int = 1, 
                 activation_function: Optional[Any] = None, # Using Any for nested class ref or Type[BaseActivationFunction]
                 input_shape: Optional[Tuple[int, ...]] = None) -> None:
        
        if neuron_count < 1:
            raise ValueError("neuron_count must be greater then 0!")
        super().__init__()
        self.shape = (neuron_count,)
        self.biases = np.zeros(shape=(neuron_count), dtype=NP_FLOAT_PRECISION) # neuron biases stored as np array
        self.name = "Dense Layer"
        self.activation_function = activation_function
        if input_shape:
            self.init_params(input_shape)


    def feedforward(self, input_array: npt.NDArray[np.float32], save: bool = False) -> npt.NDArray[np.float32]:
        """
        feeds the input data forward through layer and returns the output a numpy array
        Performs: Output = Activation(Weights * Input + Biases)
        """
        if self.weights is None:
                raise ValueError("Weights not initialized. Call init_params first.")
        
        if input_array.shape[0] != self.weights.shape[1]:
            raise ValueError(f"Input array shape {input_array.shape} doesn't match the shape of the layers weights shape {self.weights.shape}")
                    
        # Linear transformation: z = wx + b
        z = np.dot(self.weights, input_array) + self.biases
        
        # Check for numerical instability (exploding gradients/inputs)
        if not np.all(np.isfinite(z)):
            raise ValueError("NaN or Inf detected in forward pass")

        a = z

        # Apply non-linear activation if one exists
        if self.activation_function:
            a = self.activation_function.apply(z)

        # Cache the values if training (save=True)
        if save:
            self.saved = self.save_function(self.saved, input_array, z.copy(), a.copy())

        return a
    

    def feedbackwards(self, lose_derivatives: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        accepts a lose_derivatives which is ∂L/∂a derivative of lose function with respect to the 
        activation of current layer and returns ∂L/∂a(L-1), ∂L/∂w, ∂L/∂b
        """
        if self.weights is None:
            raise ValueError("Weights not initialized.")
        if self.saved is None or self.saved.prev_input is None:
            raise ValueError("No saved state for backprop. Did you run feedforward with save=True?")

        if lose_derivatives.shape != self.shape:
            raise ValueError(f"lose_derivatives {lose_derivatives.shape} doesn't equal current layer shape {self.shape}")
        
        # dL/dz starts as dL/da
        dL_dz = lose_derivatives
        
        # If there is an activation function, apply chain rule: dL/dz = dL/da * da/dz
        if self.activation_function:
            dL_dz = self.activation_function.calculate_dl_dz(lose_derivatives, self.saved)

        # dL/dw = dL/dz * dz/dw (where dz/dw is the input from the previous layer)
        dz_dw = np.tile(self.saved.prev_input, (self.shape[0],1))

        # Return tuple:
        # 1. Gradient for previous layer (dL/dx = W.T * dL/dz)
        # 2. Gradient for weights (dL/dw)
        # 3. Gradient for biases (dL/db = dL/dz)
        return np.dot(self.weights.T, dL_dz), (dz_dw.T * dL_dz).T, dL_dz

    def adjust_parameters(self, gradients: List[npt.NDArray[np.float32]]) -> None:
        # Updates internal weights/biases given gradients calculated in backprop
        if self.weights is None or self.biases is None:
                raise ValueError("Weights/Biases not initialized.")

        if len(gradients) != 2:
            raise ValueError(f"gradients are of {len(gradients)} should be of length 2 for {self.name} formatted as [weights_gradient, biases_gradient]")
        if gradients[0].shape != self.weights.shape:
            raise ValueError(f"shape for weights gradients {gradients[0].shape} should be same as weights shape {self.weights.shape}")
        if gradients[1].shape != self.biases.shape:
            raise ValueError(f"shape for biases gradients {gradients[1].shape} should be same as weights shape {self.biases.shape}")
        
        # Subtract gradients (Gradient Descent step)
        self.weights -= gradients[0].astype(NP_FLOAT_PRECISION)
        self.biases -= gradients[1].astype(NP_FLOAT_PRECISION)

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("NaN or Inf detected in weights of layer pass")
        if not np.all(np.isfinite(self.biases)):
            raise ValueError("NaN or Inf detected in biases of layer pass")
        
    def init_params(self, input_shape: Tuple[int, ...]) -> None:
        # Xavier/Glorot initialization: random normal scaled by 1/sqrt(input_size)
        self.weights = np.random.normal(size=(self.shape[0], input_shape[0])).astype(NP_FLOAT_PRECISION) * np.sqrt(1/input_shape[0]) # weights represented as 2nd numpy array rows representing the current layer neuron index and column previous inputs