import numpy as np
import tensorflow as tf
from tensorflow import keras
from omni import *
from tqdm import tqdm


(x_train_2d, y_train), (x_test_2d, y_test) = tf.keras.datasets.mnist.load_data()
# train size: 60,000
# test size: 10,000

# making 2d array of pixel values flat
x_train_flat = x_train_2d.reshape(-1, 784) / 255
x_test_flat = x_test_2d.reshape(-1, 784) / 255

tf_model = keras.models.load_model("model/tf_mnist.keras")
omni_model = Model.load("model/omni_mnist.model")

# feeding test data to the TF model
tf_y_ = tf_model.predict(x_test_flat)
tf_correct = np.sum(np.argmax(tf_y_, axis=1) == y_test)

# feeding test data to omni model
omni_correct = 0
for X, y in tqdm(zip(x_test_flat, y_test)):
    y_ = omni_model.predict(X)
    if np.argmax(y_) == y:
        omni_correct += 1
        
# calculating and showing accuracy
print("=== mnist data set accuracy ===")
print(f"TF accuracy: {tf_correct/len(y_test)*100}%")
print(f"Omni accuracy: {omni_correct/len(y_test)*100}%")