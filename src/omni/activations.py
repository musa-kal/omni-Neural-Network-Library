import numpy as np
import numpy.typing as npt
from .utils import LayerSave

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