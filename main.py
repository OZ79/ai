from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weights_count: int, f_activation_n: int = 1):
        self.weights_count = weights_count
        self.z = None
        self.output = None
        self.f_activation_n = f_activation_n
        self.wb = None
        self.momentum = None

    @property
    def f_activation(self) -> Callable[[float], float]:
        def relu(z:float) -> float:
            return max(0, z)
        def linear(z):
            return z
        
        if self.f_activation_n == 1:
            return relu
        elif self.f_activation_n == 2:
            return linear

        raise ValueError("Unknown activation")

    @property
    def f_activation_prim(self) -> Callable[[float], float]:
        def relu_grad(z:float) -> float:
                return 1.0 if z > 0 else 0.0
        def linear_prime(z):
            return 1.0
        
        if self.f_activation_n == 1:
            return relu_grad
        elif self.f_activation_n == 2:
            return linear_prime

        raise ValueError("Unknown activation")

    def init(self) -> Neuron:
        self.wb = np.concatenate([np.random.randn(self.weights_count) * np.sqrt(2 / self.weights_count),[0.0]])
        self.momentum = np.zeros_like(self.wb)
        return self

    def forward(self, input_data:np.ndarray) -> float:
        input_extended = np.concatenate([input_data, [1]])
        self.z = self.wb @ input_extended
        self.output = self.f_activation(self.z)
        return self.output

    @property
    def prim(self)-> float:
        return self.f_activation_prim(self.z)


class Layer:
    def __init__(self, inputs_count: int, neurons_count: int, f_activation_n: int = 1):
        self.inputs_count = inputs_count
        self.neurons_count = neurons_count
        self.neurons = [
            Neuron(weights_count=inputs_count, f_activation_n=f_activation_n).init() 
            for _ in range(0, self.neurons_count)
            ]
        self.output = None

    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.output = np.array([self.neurons[n].forward(input_data) for n in range(0, self.neurons_count)])
        return self.output

    @property
    def prims(self)-> np.ndarray:
        return np.array([neuron.prim for neuron in self.neurons])


class NeuralNetwork:
    def __init__(self, inputs_count:int):
        self.inputs_count = inputs_count
        self.input_data = None
        self.output = None
        self.layers: list[Layer] = []
        self.layers_count = None
        self.loss = []

    def add_layer(self, neurons_count: int, f_activation_n=1):
        if not self.layers:
            self.layers.append(Layer(self.inputs_count, neurons_count, f_activation_n))
        else:
            self.layers.append(Layer(self.layers[-1].neurons_count, neurons_count, f_activation_n))
        self.layers_count = len(self.layers)
        
    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.input_data = input_data
        output = self.input_data;
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return self.output

    def backward(self, train_data:np.ndarray, learning_rate = 0.01):
        gradients = [None] * self.layers_count
        for layerNum in range(self.layers_count -1, -1, -1):
            layer = self.layers[layerNum]
            if layerNum == self.layers_count -1:
                self.loss.append(np.sum((self.output - train_data) ** 2) * 0.5)
                delta = (self.output - train_data) * layer.prims
            else:
                layerNext =  self.layers[layerNum + 1]
                delta = (np.array([neuron.wb[:-1] for neuron in layerNext.neurons]).T  @  delta) * layer.prims
            if layerNum == 0:
                gradient = np.outer(delta, np.concatenate([self.input_data, [1]]))
            else:
                gradient = np.outer(delta, np.concatenate([self.layers[layerNum -1].output, [1]]))

            gradients[layerNum] = gradient

        for layerNum in range(self.layers_count):
            self.updateWb(
                self.layers[layerNum], gradients[layerNum], learning_rate)

    def updateWb(self, layer: Layer, gradients, learning_rate):
        momentum = 0.9
        max_norm = 1.0

        for i, neuron in enumerate(layer.neurons):

            gradient = gradients[i]

            norm = np.linalg.norm(gradient)

            if norm > max_norm:
                gradient = gradient * (max_norm / norm)

            velocity = (
                momentum * neuron.momentum
                + learning_rate * gradient
            )

            neuron.wb -= velocity
            neuron.momentum = velocity


######################################################################

nn = NeuralNetwork(2)
nn.add_layer(6)
nn.add_layer(6)
nn.add_layer(6)
nn.add_layer(1,2)

train_X = []
train_y = []

for _ in range(1500):
    x1 = np.random.randint(-20, 21)
    x2 = np.random.randint(-20, 21)

    train_X.append(
        np.array([x1, x2])
    )

    train_y.append(
        np.array([
            x1 * x2
        ])
    )

train_X = np.array(train_X, dtype=float)
train_y = np.array(train_y, dtype=float)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

train_X_scaled = X_scaler.fit_transform(train_X)
train_y_scaled = y_scaler.fit_transform(train_y)

epoch_losses = []

plt.ion()

fig, ax = plt.subplots()

line, = ax.plot([], [])

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training loss")
ax.grid()

try:
    for epoch in range(3000):

        epoch_loss = 0.0

        indices = np.random.permutation(len(train_X_scaled))
        for i in indices:
            x = train_X_scaled[i]
            y = train_y_scaled[i]

            nn.forward(x)
            nn.backward(y, learning_rate=0.001)

            epoch_loss += nn.loss[-1]

        epoch_loss /= len(train_X)
        epoch_losses.append(epoch_loss)

        if epoch % 100 == 0:
            line.set_xdata(range(len(epoch_losses)))
            line.set_ydata(epoch_losses)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

except KeyboardInterrupt:
    print("\nНавчання достроково завершено!")

plt.ioff()
plt.show()

y_pred = []
y_true = []

test_X = []
test_y = []

for _ in range(100):
    x1 = np.random.randint(-20, 20)
    x2 = np.random.randint(-20, 20)

    test_X.append(
        np.array([x1, x2], dtype=float)
    )

    test_y.append(
        np.array([x1 * x2], dtype=float)
    )

for x, y in zip(test_X, test_y):

    # масштабування входу
    x_scaled = X_scaler.transform(
        x.reshape(1, -1)
    )

    # прогноз мережі
    result_scaled = nn.forward(x_scaled[0])

    # повертаємо prediction до початкового масштабу
    result = y_scaler.inverse_transform(
        result_scaled.reshape(1, -1)
    )[0]

    y_pred.append(result)
    y_true.append(y)

y_pred = np.array(y_pred)
y_true = np.array(y_true)

r2 = r2_score(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("R² =", r2)
print("rmse =", rmse)
print("MAE =", mae)

results = []

for _ in range(20):
    x1 = np.random.randint(-20, 20)
    x2 = np.random.randint(-20, 20)

    x = np.array([[x1, x2]], dtype=float)

    x_scaled = X_scaler.transform(x)
    result_scaled = nn.forward(x_scaled[0])

    result = y_scaler.inverse_transform(
        result_scaled.reshape(1, -1)
    )[0, 0]

    results.append({
        'x1': x1,
        'x2': x2,
        'NN result': result,
        'x1 * x2': x1 * x2
    })

df = pd.DataFrame(results)

print(df)