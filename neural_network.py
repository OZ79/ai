from __future__ import annotations
from enum import Enum
from typing import Callable
import numpy as np

# ============================================================
# NEURON
# ============================================================

class ActivationFunc(Enum):
    Relu = "Relu"
    Linear = "Linear"

class Neuron:
    def __init__(self, weights_count: int, f_activation_n: ActivationFunc = ActivationFunc.Linear):
        self.weights_count = weights_count
        self.z = None
        self.output = None
        self.f_activation_n = f_activation_n
        self.wb = None
        self.momentum = None

    @property
    def f_activation(self) -> Callable[[float], float]:
        def linear(z):
            return z
        def relu(z:float) -> float:
            return max(0, z)

        if self.f_activation_n == ActivationFunc.Linear:
            return linear
        elif self.f_activation_n == ActivationFunc.Relu:
            return relu

        raise ValueError("Unknown activation")

    @property
    def f_activation_prim(self) -> Callable[[float], float]:
        def linear_prime(z):
            return 1.0
        def relu_grad(z:float) -> float:
            return 1.0 if z > 0 else 0.0
        
        if self.f_activation_n == ActivationFunc.Linear:
            return linear_prime
        elif self.f_activation_n == ActivationFunc.Relu:
            return relu_grad

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

# ============================================================
# LAYER
# ============================================================

class Layer:
    def __init__(self, inputs_count: int, neurons_count: int, f_activation_n: ActivationFunc = ActivationFunc.Linear):
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

# ============================================================
# NEURAL NETWORK
# ============================================================

class NeuralNetwork:
    def __init__(self, inputs_count:int):
        self.inputs_count = inputs_count
        self.input_data = None
        self.output = None
        self.layers: list[Layer] = []
        self.layers_count = None
        self.loss = []

    def add_layer(self, neurons_count: int, f_activation_n: ActivationFunc = ActivationFunc.Linear):
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
                input = np.concatenate([self.input_data, [1]])
            else:
                input = np.concatenate([self.layers[layerNum -1].output, [1]])
            gradient = np.outer(delta, input)
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

