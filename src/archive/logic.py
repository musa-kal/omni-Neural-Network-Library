"""
Author: Musa Kaleem
"""

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import List, Tuple, Optional, Union, Any

# Set the standard float precision for the library to ensure consistency
NP_FLOAT_PRECISION = np.float32

class LayerSave:
    """
    A helper class used to cache values during the forward pass.
    These values (input, pre-activation Z, post-activation A) are required 
    later during backpropagation to calculate gradients.
    """
    def __init__(self, 
                 prev_input: Optional[npt.NDArray[np.float32]] = None, 
                 pre_activation: Optional[npt.NDArray[np.float32]] = None, 
                 post_activation: Optional[npt.NDArray[np.float32]] = None) -> None:
        self.prev_input: Optional[npt.NDArray[np.float32]] = prev_input         # The input x fed into the layer
        self.pre_activation: Optional[npt.NDArray[np.float32]] = pre_activation # The linear result z = wx + b
        self.post_activation: Optional[npt.NDArray[np.float32]] = post_activation # The result after activation a = f(z)

    def __repr__(self) -> str:
        return f"pre_input: {self.prev_input}\npre_activation: {self.pre_activation}\npost_activation: {self.post_activation}"

class ActivationFunctions:
    """
    Activation Functions class will hold different pre built activation function.
    Contains logic for both the forward pass (apply) and backward pass (derivative).
    """

    class BaseActivationFunction:

        name: str = "Base"
        
        @staticmethod
        def apply(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Computes f(z)
            raise NotImplementedError(f"apply must be implemented in child class!")
        
        @staticmethod
        def derivative(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Computes f'(z)
            raise NotImplementedError(f"derivative must be implemented in child class!")
        
        @staticmethod
        def calculate_dl_dz(dl_da: npt.NDArray[np.float32], saved: LayerSave) -> npt.NDArray[np.float32]:
            # Computes chain rule: dL/dz = dL/da * da/dz
            raise NotImplementedError(f"calculate_dl_dz must be implemented in child class!")


    class Relu(BaseActivationFunction):

        name = "Relu"
        
        @staticmethod
        def apply(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Rectified Linear Unit: returns z if z > 0, else 0
            return np.maximum(z, 0)
        
        @staticmethod
        def derivative(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Derivative is 1 if z > 0, else 0
            return np.where(z < 0, 0, 1) # type: ignore
        
        @staticmethod
        def calculate_dl_dz(dl_da: npt.NDArray[np.float32], saved: LayerSave) -> npt.NDArray[np.float32]:
            # Element-wise multiplication for chain rule
            if saved.pre_activation is None:
                raise ValueError("Pre-activation not saved for Relu backward pass")
            return ActivationFunctions.Relu.derivative(saved.pre_activation) * dl_da
    
    class Softmax(BaseActivationFunction):

        name = "Softmax"

        @staticmethod
        def apply(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Subtract max(z) for numerical stability (prevents overflow in exp)
            ez = np.exp(z - np.max(z))
            return ez / np.sum(ez)
        
        @staticmethod
        def derivative(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            """
            equivalent to S[j] (∂[i,j] - S[i])
            where ∂[i,j] = 1 if i == j else 0
            Computes the Jacobian matrix because Softmax output depends on all inputs.
            """
            s = z.reshape(-1,1)
            return np.diagflat(s) - np.dot(s, s.T)
        
        @staticmethod
        def calculate_dl_dz(dl_da: npt.NDArray[np.float32], saved: LayerSave) -> npt.NDArray[np.float32]:
            # Optimization for Softmax combined with Cross-Entropy usually simplifies,
            # but this is the raw chain rule application for Softmax gradient.
            if saved.post_activation is None:
                raise ValueError("Post-activation not saved for Softmax backward pass")
            s = np.dot(saved.post_activation, dl_da)
            return saved.post_activation * (dl_da - s)
        
    class Sigmoid(BaseActivationFunction):
        
        name = "Sigmoid"
        
        @staticmethod
        def apply(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # S-curve function mapping inputs to (0, 1)
            return 1 / (1 + np.exp(-z))
        
        @staticmethod
        def derivative(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Derivative property of sigmoid: f'(z) = f(z)(1 - f(z))
            a = ActivationFunctions.Sigmoid.apply(z)
            return a * (1 - a)
        
        @staticmethod
        def calculate_dl_dz(dl_da: npt.NDArray[np.float32], saved: LayerSave) -> npt.NDArray[np.float32]:
            # dL/dz = a * (1 - a) * dL/da
            if saved.post_activation is None:
                 raise ValueError("Post-activation not saved for Sigmoid backward pass")
            return saved.post_activation * (1 - saved.post_activation) * dl_da
        


class Sequential:
    """
    Layers class holds logic for joining and infracting with joint layers and some other helper classes.
    Essentially acts as a container or 'Sequential' model wrapper for individual layers.
    """

    class BaseLayer:
        """
        Base Layer class used to create different types of hidden layers
        """
        
        def __init__(self) -> None:
            self.shape: Tuple[int, ...] = ()
            self.name: Optional[str] = None
            self.saved: Optional[LayerSave] = None # Holds the LayerSave object (cache) for this layer
            self.weights: Optional[npt.NDArray[np.float32]] = None
            self.biases: Optional[npt.NDArray[np.float32]] = None
            
            def save_function(prev_save: Optional[LayerSave], 
                              prev_input: Optional[npt.NDArray[np.float32]], 
                              pre_activation: Optional[npt.NDArray[np.float32]], 
                              post_activation: Optional[npt.NDArray[np.float32]]) -> LayerSave:
                return LayerSave(prev_input, pre_activation, post_activation)
                
            self.save_function = save_function


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
        
        def init_weights(self, input_shape: Tuple[int, ...]) -> None:
            # Initializes weights based on input size
            raise NotImplementedError(f"init_weights must be implemented in the child class {self.name}!")
 

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
                self.init_weights(input_shape)


        def feedforward(self, input_array: npt.NDArray[np.float32], save: bool = False) -> npt.NDArray[np.float32]:
            """
            feeds the input data forward through layer and returns the output a numpy array
            Performs: Output = Activation(Weights * Input + Biases)
            """
            if self.weights is None:
                 raise ValueError("Weights not initialized. Call init_weights first.")
            
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
            
        def init_weights(self, input_shape: Tuple[int, ...]) -> None:
            # Xavier/Glorot initialization: random normal scaled by 1/sqrt(input_size)
            self.weights = np.random.normal(size=(self.shape[0], input_shape[0])).astype(NP_FLOAT_PRECISION) * np.sqrt(1/input_shape[0]) # weights represented as 2nd numpy array rows representing the current layer neuron index and column previous inputs





    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape: Tuple[int, ...] = input_shape
        self.layers: List[Sequential.BaseLayer] = []        

    def join_front(self, new_layer: BaseLayer) -> None:
        """
        takes a BaseLayer class and joins it to the front/right of the network.
        Essentially 'append' a layer to the stack.
        """

        if len(self.layers) == 0:
            # First layer must match the network's input shape
            if len(new_layer.shape) != len(self.input_shape):
                raise ValueError(f"new layer shape {new_layer.shape} doesn't match the input shape {self.input_shape}!")
            new_layer.init_weights(self.input_shape)
            self.layers.append(new_layer)
            return
        
        # Subsequent layers must match the output shape of the previous layer
        if len(new_layer.shape) != len(self.layers[-1].shape):
            raise ValueError(f"new layer shape {new_layer.shape} doesn't match the previous layer {self.layers[-1].shape}!")
        new_layer.init_weights(self.layers[-1].shape)
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
    
class Model:
    """
    High-level class to compile, train (fit), and use (predict) the neural network.
    """
    
    class LoseFunction:
        name = "Base Lose Function"
        
        @staticmethod
        def apply(y: npt.NDArray[np.float32], _y: npt.NDArray[np.float32]) -> float:
            raise NotImplementedError("apply must be implemented in child class!")
        
        @staticmethod
        def derivative(y: npt.NDArray[np.float32], _y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            raise NotImplementedError("derivative must be implemented in child class!")
        
    class MSE(LoseFunction):
        # Mean Squared Error: sum((y_pred - y_true)^2)
        name = "MSE"
        
        @staticmethod
        def apply(y: npt.NDArray[np.float32], _y: npt.NDArray[np.float32]) -> float:
            return np.sum((_y - y)**2) # type: ignore
        
        @staticmethod
        def derivative(y: npt.NDArray[np.float32], _y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Gradient of MSE
            return 2 * (_y - y)
        
    class CrossEntropy(LoseFunction):
        # Cross Entropy Loss (often used with Softmax)
        name = "Cross Entropy"
        
        @staticmethod
        def apply(y: npt.NDArray[np.float32], _y: npt.NDArray[np.float32]) -> float:
            return -np.sum(y * np.log(_y)) # type: ignore
        
        @staticmethod
        def derivative(y: npt.NDArray[np.float32], _y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            return -y / _y
        
    
    def __init__(self, layers: Sequential) -> None:
        self.layers = layers
        self.alpha: float = 0.0 # Learning rate
        self.loss_function: Optional[Any] = None # Using Any for LoseFunction class type
    
    def compile(self, alpha: float = 0.001, loss_function: Any = MSE) -> None:
        # Sets hyperparameters before training
        self.alpha = alpha
        self.loss_function = loss_function
    
    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.float32], batch_size: int = 1, epoch: int = 1) -> None:
        """
        Main training loop using Stochastic Gradient Descent (SGD) with mini-batches.
        """
        n = len(X)
        
        if n != len(y):
            raise ValueError(f"size of y {len(y)} should be same as X {n}")
        if n < batch_size:
            raise ValueError(f"batch size {batch_size} should be less then X: {n} samples")
        
        if self.loss_function is None:
            raise ValueError("Model not compiled. Call model.compile() before fitting.")

        # Iterate over the dataset 'epoch' times
        for curr_itr in range(epoch):
            # Shuffle data at the start of every epoch to prevent cycles
            idxs = np.random.permutation(n)
            shuffled_X = X[idxs]
            shuffled_y = y[idxs]
            t_loss = 0.0
            
            # Mini-batch training loop
            for batch_group_i in tqdm(range(0, n, batch_size), desc="Training Progress", dynamic_ncols=True):
                X_batch = shuffled_X[batch_group_i: batch_group_i+batch_size]
                y_batch = shuffled_y[batch_group_i: batch_group_i+batch_size]
                
                grad_sum: Optional[List[List[npt.NDArray[np.float32]]]] = None # Accumulator for gradients within a batch
                
                # Process each sample in the batch individually
                for input_X, expected_y in zip(X_batch, y_batch):
                                                    
                    # 1. Forward Pass (with save=True to cache values)
                    predicted_y = self.layers.feedforward(input_data=input_X, save=True)
                    
                    if expected_y.shape != predicted_y.shape:
                        raise ValueError(f"value in provided y has a shape of {expected_y.shape} which doesn't match the output shape of {predicted_y.shape}")
                    
                    # 2. Calculate Loss
                    t_loss += self.loss_function.apply(expected_y, predicted_y)

                    # 3. Backward Pass (Calculate Gradients)
                    # propagate_backwards receives the derivative of Loss w.r.t Output
                    loss_deriv = self.loss_function.derivative(expected_y, predicted_y)
                    grads = self.layers.propagate_backwards(loss_deriv)
                    
                    # 4. Accumulate Gradients
                    if grad_sum is not None:
                        for grad_i, grad in enumerate(grads):
                            for grad_j in range(len(grad)):
                                grad_sum[grad_i][grad_j] += grad[grad_j]
                    else:
                        grad_sum = grads
                
                if grad_sum is None:
                    continue

                # 5. Update Weights (Average gradients over batch * Learning Rate)
                for grad_sum_i, curr_grad_sum in enumerate(grad_sum):
                    for grad_sum_j in range(len(curr_grad_sum)):
                        # Scale accumulated gradients by (1/batch_size) and learning rate (alpha)
                        grad_sum[grad_sum_i][grad_sum_j] = curr_grad_sum[grad_sum_j] / batch_size * self.alpha
                    
                    # Apply the update to the specific layer
                    self.layers.adjust_layer_parameter(grad_sum_i, grad_sum[grad_sum_i])
                
            tqdm.write(f"Epoch #{curr_itr+1}/{epoch} - total loss/samples: {t_loss/n}")
                
                
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Public method to get predictions for new data (no saving/caching)
        return self.layers.feedforward(X)
        

if __name__ == '__main__':
        
    print("===simple model test===")
    
    np.random.seed(12)
    # Generate random training data
    X: npt.NDArray[np.float32] = (2 * np.random.rand(100, 1)).astype(NP_FLOAT_PRECISION)
    # Generate target values (Linear relation: y = 4x + 5 + noise)
    y: npt.NDArray[np.float32] = (5 + 4 * X + np.random.randn(100, 1)).astype(NP_FLOAT_PRECISION)
    
    # Initialize Layer container
    x = Layers(input_shape=(1,))
    # Add a Dense layer with 1 neuron (Linear Regression)
    x.join_front(Layers.DenseLayer(1))
    
    # Compile and train model
    model = Model(x)
    model.compile(loss_function=Model.MSE)
    model.fit(X, y, epoch=50, batch_size=10)
    
    print(model.layers)