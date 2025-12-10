import numpy as np
import numpy.typing as npt
from typing import Optional

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