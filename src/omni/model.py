import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import List, Optional, Any
from .network import Sequential
import pickle
import os

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
    
    
    def save(self, file_path):
        """
        Instance method to save the current model instance to a file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved at {file_path}")
        
    @classmethod
    def load(cls, file_path):
        """
        Class method to load a model a saved model.
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
            
        with open(file_path, 'rb') as f:
            instance = pickle.load(f)
            print(f"Loaded model from {file_path}")
            return instance