import numpy as np
import tensorflow as tf
from tensorflow import keras
from omni import *
from sklearn.preprocessing import OneHotEncoder


(x_train_2d, y_train), (x_test_2d, y_test) = tf.keras.datasets.mnist.load_data()
# train size: 60,000
# test size: 10,000

# making 2d array of pixel values flat
x_train_flat = x_train_2d.reshape(-1, 784) / 255
x_test_flat = x_test_2d.reshape(-1, 784) / 255

tmodel = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

tmodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

history = tmodel.fit(
    x=x_train_flat,
    y=y_train,
    epochs=5,
    batch_size=32,
)

tmodel.save('model/tf_model.keras')

# turning labels into one hot vector encodings to train omni model 
encoder = OneHotEncoder(categories=[range(10)], sparse_output=False)
one_hot_vectors_train = encoder.fit_transform(y_train.reshape(-1, 1))

x = Layers(input_shape=(784,))
x.join_front(DenseLayer(128, ActivationFunctions.Relu))
x.join_front(DenseLayer(10, ActivationFunctions.Softmax))
omni_model = Model(x)
omni_model.compile(loss_function=Model.CrossEntropy)
omni_model.fit(x_train_flat, 
               one_hot_vectors_train, 
               batch_size=32, 
               epoch=5)

omni_model.save("model/omni_mnist.md")