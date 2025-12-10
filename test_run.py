import numpy as np
from omni import DenseLayer, Model, ActivationFunctions, Layers

print("===model===")

# np.random.seed(12)
X = 2 * np.random.rand(100, 1)
y = 100 * (X - 1) ** 2 + np.random.randn(100, 1)

x = Layers(input_shape=(1,))
x.join_front(DenseLayer(64, ActivationFunctions.Relu))
x.join_front(DenseLayer(64, ActivationFunctions.Relu))
x.join_front(DenseLayer(1))
model = Model(x)
model.compile(loss_function=model.MSE)
model.fit(X, y, epoch=300, batch_size=16)

from tensorflow import keras

tmodel = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

tmodel.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mse'
)

tmodel.fit(
    X, y,
    epochs=300,
    batch_size=16
)


import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Data Points: 100 * (X - 1) ** 2 + ε')
plt.scatter(X, tuple(model.predict(x) for x in X), color='red', label='Omni fit line')
plt.scatter(X, tuple(tmodel.predict(x) for x in X), color='green', label='TF fit line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
    
    