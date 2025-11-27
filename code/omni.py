"""
Author: Musa Kaleem
"""

import numpy as np
import numpy.typing as npt

NP_FLOAT_PRECISION = np.float32

class LayerSave:
    def __init__(self, prev_input=None, pre_activation=None, post_activation=None):
        self.prev_input = prev_input
        self.pre_activation = pre_activation
        self.post_activation = post_activation

    def __repr__(self):
        return f"pre_input: {self.prev_input}\npre_activation: {self.pre_activation}\npost_activation: {self.post_activation}"

class ActivationFunctions:
    """
    Activation Functions class will hold different pre built activation function.
    """

    class BaseActivationFunction:

        name = None
        
        @staticmethod
        def apply(z: npt.NDArray):
            raise NotImplementedError(f"apply must be implemented in child class!")
        
        @staticmethod
        def derivative(z: npt.NDArray):
            raise NotImplementedError(f"derivative must be implemented in child class!")
        
        @staticmethod
        def calculate_dl_dz(dl_da: npt.NDArray, saved: LayerSave):
            raise NotImplementedError(f"calculate_dl_dz must be implemented in child class!")


    class Relu(BaseActivationFunction):

        name = "Relu"
        
        @staticmethod
        def apply(z):
            return np.maximum(z, 0)
        
        @staticmethod
        def derivative(z):
            return np.where(z < 0, 0, 1)
        
        @staticmethod
        def calculate_dl_dz(dl_da, saved):
            return ActivationFunctions.Relu.derivative(saved.pre_activation) * dl_da
    
    class Softmax(BaseActivationFunction):

        name = "Softmax"

        @staticmethod
        def apply(z):
            ez = np.exp(z-np.max(z))
            return ez / np.sum(ez)
        
        @staticmethod
        def derivative(z):
            """
            equivalent to S[j] (∂[i,j] - S[i])
            where ∂[i,j] = 1 if i == j else 0
            """
            s = z.reshape(-1,1)
            return np.diagflat(s) - np.dot(s, s.T)
        
        @staticmethod
        def calculate_dl_dz(dl_da, saved):
            s = np.dot(saved.post_activation, dl_da)
            return saved.post_activation * (dl_da - s)


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
            self.saved = None
            
            def save_function(prev_save: LayerSave, prev_input: npt.ArrayLike, pre_activation: npt.ArrayLike, post_activation: npt.ArrayLike):
                return LayerSave(prev_input, pre_activation, post_activation)
                
            self.save_function = save_function


        def feedforward(self, input_array: npt.NDArray):
            """
            should return a numpy array
            """
            raise NotImplementedError(f"feedforward must be implemented in the child class {self.name}!")
        
        def feedbackwards(self, loss: npt.NDArray):
            """
            still figuring out how the returns should work
            """
            raise NotImplementedError(f"feedbackwards must be implemented in the child class {self.name}!")
        

        def clear_all_saved_data(self):
            """
            helper function to clear all saved data within the current layer
            """
            self.saved = None
        
        def adjust_parameters(self, gradients):
            raise NotImplementedError(f"adjust_parameters must be implemented in the child class {self.name}!")
 
        

        def __repr__(self):
            """
            return string including the layer name and shape
            """
            return f"{self.name}-{self.shape}"


    class DenseLayer(BaseLayer):
        """
        Dense Layer class inherits from BaseLayer and represents a dense layer
        """
        def __init__(self, neuron_count:int=1, activation_function:ActivationFunctions.BaseActivationFunction|None=None, input_shape:tuple=(1,)):
            if neuron_count < 1:
                raise ValueError("neuron_count must be greater then 0!")
            super().__init__()
            self.shape = (neuron_count,)
            self.biases = np.zeros(shape=(neuron_count), dtype=NP_FLOAT_PRECISION) # neuron biases stored as np array
            self.name = "Dense Layer"
            self.activation_function = activation_function
            self.weights = np.random.normal(size=(neuron_count, input_shape[0])).astype(NP_FLOAT_PRECISION) * np.sqrt(1/input_shape[0]) # weights represented as 2nd numpy array rows representing the current layer neuron index and column previous inputs
        

        def feedforward(self, input_array: npt.NDArray, save=False):
            """
            feeds the input data forward through layer and returns the output a numpy array
            """
            if input_array.shape[0] != self.weights.shape[1]:
                raise ValueError(f"Input array shape {input_array.shape} doesn't match the shape of the layers weights shape {self.weights.shape}")
                        
            z = np.dot(self.weights, input_array) + self.biases
            
            if not np.all(np.isfinite(z)):
                raise ValueError("NaN or Inf detected in forward pass")

            a = z

            if self.activation_function:
                a = self.activation_function.apply(z)

            if save:
                self.saved = self.save_function(self.saved, input_array, z.copy(), a.copy())

            return a
        

        def feedbackwards(self, lose_derivatives: npt.NDArray):
            """
            accepts a lose_derivatives which is ∂L/∂a derivative of lose function with respect to the 
            activation of current layer and returns ∂L/∂a(L-1), ∂L/∂w, ∂L/∂b
            """
            if lose_derivatives.shape != self.shape:
                raise ValueError(f"lose_derivatives {lose_derivatives.shape} doesn't equal current layer shape {self.shape}")
            
            dL_dz = lose_derivatives
            
            if self.activation_function:
                dL_dz = self.activation_function.calculate_dl_dz(lose_derivatives, self.saved)

            dz_dw = np.tile(self.saved.prev_input, (self.shape[0],1))

            # ∂L/∂a(L-1), ∂L/∂w, ∂L/∂b
            return np.dot(self.weights.T, dL_dz), (dz_dw.T * dL_dz).T, dL_dz

        def adjust_parameters(self, gradients: list[npt.DTypeLike]):
            if len(gradients) != 2:
                raise ValueError(f"gradients are of {len(gradients)} should be of length 2 for {self.name} formatted as [weights_gradient, biases_gradient]")
            if gradients[0].shape != self.weights.shape:
                raise ValueError(f"shape for weights gradients {gradients[0].shape} should be same as weights shape {self.weights.shape}")
            if gradients[1].shape != self.biases.shape:
                raise ValueError(f"shape for biases gradients {gradients[1].shape} should be same as weights shape {self.biases.shape}")
            
            self.weights -= gradients[0].astype(NP_FLOAT_PRECISION)
            self.biases -= gradients[1].astype(NP_FLOAT_PRECISION)

            if not np.all(np.isfinite(self.weights)):
                raise ValueError("NaN or Inf detected in weights of layer pass")
            if not np.all(np.isfinite(self.biases)):
                print(self.biases)
                raise ValueError("NaN or Inf detected in biases of layer pass")





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


    def feedforward(self, input_data: npt.NDArray, save=False):
        """
        takes input data and feeds it through the network returning the output.
        """
        if input_data.shape != self.input_shape:
            raise ValueError(f"input_data shape {input_data.shape} doesn't match the shape specified f{self.input_shape}")
        

        next_input = input_data
        for layer in self.layers:
            curr_output = layer.feedforward(next_input, save)
            next_input = curr_output
        
        return next_input

    def propagate_backwards(self, layer_output: npt.NDArray):
        
        layer_derivate = []
        next_input = layer_output

        for layer in self.layers[::-1]:
            output = layer.feedbackwards(next_input)
            layer_derivate.append([output[1], output[2]])
            next_input = output[0]

        return layer_derivate[::-1]
    
    def adjust_layer_parameter(self, i:int, gradients:list[npt.NDArray]):
        if i < 0 or i > len(self.layers):
            raise ValueError(f"i must be with in range ({0},{len(self.layers)}]")
        self.layers[i].adjust_parameters(gradients) 

    def __repr__(self):
        """
        string representing the network.
        """
        output = f"layers: [Input-{self.input_shape}\t" 
        for layer in self.layers:
            output += f"{layer}\t"
        return output.strip() + "]"
    
class Model:
    
    class LoseFunction:
        name = "Base Lose Function"
        
        @staticmethod
        def apply(y: npt.ArrayLike, _y: npt.ArrayLike):
            raise NotImplementedError("apply must be implemented in child class!")
        
        @staticmethod
        def derivative(y: npt.ArrayLike, _y: npt.ArrayLike):
            raise NotImplementedError("derivative must be implemented in child class!")
        
    class MSE(LoseFunction):
        name = "MSE"
        
        @staticmethod
        def apply(y: npt.ArrayLike, _y: npt.ArrayLike):
            return np.sum((_y-y)**2)
        
        @staticmethod
        def derivative(y: npt.ArrayLike, _y: npt.ArrayLike):
            return 2*(_y-y)
        
    class CrossEntropy(LoseFunction):
        name = "Cross Entropy"
        
        @staticmethod
        def apply(y: npt.ArrayLike, _y: npt.ArrayLike):
            return -np.sum(y*np.log(_y))
        
        @staticmethod
        def derivative(y: npt.ArrayLike, _y: npt.ArrayLike):
            return -y/_y
        
    
    def __init__(self, layers: Layers):
        self.layers = layers
        self.alpha = 0
        self.loss_function = None
    
    def compile(self, alpha=0.001, loss_function=MSE):
        self.alpha = alpha
        self.loss_function = loss_function
    
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, batch_size=1, epoch=1):
        n = len(X)
        
        if n != len(y):
            raise ValueError(f"size of y {len(y)} should be same as X {n}")
        if n < batch_size:
            raise ValueError(f"batch size {batch_size} should be less then X: {n} samples")
        
        for curr_itr in range(epoch):
            idxs = np.random.permutation(n)
            shuffled_X = X[idxs]
            shuffled_y = y[idxs]
            t_loss = 0
            
            for batch_group_i in range(0, n, batch_size):
                X_batch = shuffled_X[batch_group_i: batch_group_i+batch_size]
                y_batch = shuffled_y[batch_group_i: batch_group_i+batch_size]
                
                grad_sum = None
                
                for input_X, expected_y in zip(X_batch, y_batch):
                                        
                    predicted_y = self.layers.feedforward(input_data=input_X, save=True)
                    
                    if expected_y.shape != predicted_y.shape:
                        raise ValueError(f"value in provided y has a shape of {expected_y.shape} which doesn't match the output shape of {predicted_y.shape}")
                    
                    t_loss += self.loss_function.apply(expected_y, predicted_y)
                    grads = self.layers.propagate_backwards(self.loss_function.derivative(expected_y, predicted_y))
                    
                    if grad_sum:
                        for grad_i, grad in enumerate(grads):
                            for grad_j in range(len(grad)):
                                grad_sum[grad_i][grad_j] += grad[grad_j]
                    else:
                        grad_sum = grads
                
                for grad_sum_i, curr_grad_sum in enumerate(grad_sum):
                    for grad_sum_j in range(len(curr_grad_sum)):
                        grad_sum[grad_sum_i][grad_sum_j] = curr_grad_sum[grad_sum_j] / batch_size * self.alpha
                    self.layers.adjust_layer_parameter(grad_sum_i, grad_sum[grad_sum_i])
                
            print(f"Epoch #{curr_itr+1} - total loss / samples: {t_loss/n}")
                
                
    def predict(self, X: npt.ArrayLike):
        return self.layers.feedforward(X)
        

if __name__ == '__main__':
        
    print("===model==")
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    x = Layers(input_shape=(1,))
    l = Layers.DenseLayer(1)
    x.join_front(l)
    model = Model(x)
    model.compile(loss_function=model.MSE, alpha=0.1)
    model.fit(X, y, epoch=100, batch_size=10)
    print(model.predict(np.array([2.3514])))
    print(model.layers.layers[0].weights, model.layers.layers[0].biases)
    
    import matplotlib.pyplot as plt
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, np.c_[np.ones(X.shape[0]), X].dot(
        [model.layers.layers[0].biases[0], model.layers.layers[0].weights[0][0]]), color='red', label='fit line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    
    pass
