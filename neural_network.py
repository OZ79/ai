from __future__ import annotations
from enum import Enum
from typing import Callable
import numpy as np

# NEURON

class ActivationFunc(Enum):
    Relu = "relu"
    Linear = "linear"
    Sigmoid = "sigmoid"

def linear(z):
    return z

def relu(z:float) -> float:
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear_derivative(z):
    return 1.0

def relu_derivative(z:float) -> float:
    return 1.0 if z > 0 else 0.0
            
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Neuron:
    def __init__(self, weights_count: int, activation_fun: ActivationFunc = ActivationFunc.Linear):
        self.weights_count = weights_count
        self.z = None
        self.output = None
        self.activation_fun = activation_fun
        self.wb = None
        self.momentum = None

    @property
    def f_activation(self) -> Callable[[float], float]:
        if self.activation_fun == ActivationFunc.Linear:
            return linear
        elif self.activation_fun == ActivationFunc.Relu:
            return relu
        elif self.activation_fun == ActivationFunc.Sigmoid:
            return sigmoid

        raise ValueError("Unknown activation")

    @property
    def f_activation_derivative(self) -> Callable[[float], float]:
        if self.activation_fun == ActivationFunc.Linear:
            return linear_derivative
        elif self.activation_fun == ActivationFunc.Relu:
            return relu_derivative
        elif self.activation_fun == ActivationFunc.Sigmoid:
            return sigmoid_derivative

        raise ValueError("Unknown activation")

    def init(self) -> Neuron:
        if self.activation_fun == ActivationFunc.Relu:
            scale = np.sqrt(2 / self.weights_count)
        else:
            scale = np.sqrt(1 / self.weights_count)

        self.wb = np.concatenate([np.random.randn(self.weights_count) * scale, [0.0]])
        self.momentum = np.zeros_like(self.wb)

        return self

    def forward(self, input_data:np.ndarray) -> float:
        input_extended = np.concatenate([input_data, [1]])
        self.z = self.wb @ input_extended
        self.output = self.f_activation(self.z)
        return self.output

    @property
    def derivative(self)-> float:
        return self.f_activation_derivative(self.z)

# LAYER

class Layer:
    def __init__(self, inputs_count: int, neurons_count: int, activation_fun: ActivationFunc = ActivationFunc.Linear):
        self.inputs_count = inputs_count
        self.neurons_count = neurons_count
        self.activation_fun = activation_fun
        self.neurons = [
            Neuron(weights_count=inputs_count, activation_fun=self.activation_fun ).init() 
            for _ in range(0, self.neurons_count)
            ]
        self.output = None

    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.output = np.array([self.neurons[n].forward(input_data) for n in range(0, self.neurons_count)])
        return self.output

    @property
    def derivatives(self)-> np.ndarray:
        return np.array([neuron.derivative for neuron in self.neurons])

# NEURAL NETWORK

class LossFunc(Enum):
    MSE = "Mean Squared Error"
    BCE = "Binary Cross-Entropy"

class NeuralNetwork:
    def __init__(self, inputs_count:int):
        self.inputs_count = inputs_count
        self.input_data = None
        self.output = None
        self.layers: list[Layer] = []
        self.layers_count = None
        self.loss = []

    def add_layer(self, neurons_count: int, activation_fun: ActivationFunc = ActivationFunc.Linear):
        if not self.layers:
            self.layers.append(Layer(self.inputs_count, neurons_count, activation_fun))
        else:
            self.layers.append(Layer(self.layers[-1].neurons_count, neurons_count, activation_fun))
        self.layers_count = len(self.layers)
        
    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.input_data = input_data
        output = self.input_data
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return self.output

    def backward(self, train_data:np.ndarray, learning_rate = 0.01, lossFunc : LossFunc = LossFunc.MSE):
        gradients = [None] * self.layers_count
        for layerNum in range(self.layers_count -1, -1, -1):
            layer = self.layers[layerNum]
            if layerNum == self.layers_count -1:
                if lossFunc == LossFunc.MSE:
                    self.loss.append(np.mean((self.output - train_data) ** 2)* 0.5)
                    delta = (self.output - train_data) * layer.derivatives
                elif lossFunc == LossFunc.BCE:
                    if layer.activation_fun != ActivationFunc.Sigmoid:
                        raise ValueError("BCE requires Sigmoid activation in the output layer")

                    eps = 1e-15
                    prediction = self.output
                    clipped_prediction = np.clip(prediction, eps, 1 - eps)
                    loss = -(train_data * np.log(clipped_prediction) + (1 - train_data) * np.log(1 - clipped_prediction))
                    self.loss.append(np.mean(loss))
                    # Sigmoid + BCE
                    delta = (prediction - train_data)/ prediction.size
            else:
                layerNext =  self.layers[layerNum + 1]
                delta = (np.array([neuron.wb[:-1] for neuron in layerNext.neurons]).T  @  delta) * layer.derivatives
            if layerNum == 0:
                input_data = np.concatenate([self.input_data, [1]])
            else:
                input_data = np.concatenate([self.layers[layerNum -1].output, [1]])
            gradient = np.outer(delta, input_data)
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

            velocity = momentum * neuron.momentum + learning_rate * gradient

            neuron.wb -= velocity
            neuron.momentum = velocity

