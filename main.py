from __future__ import annotations
from typing import Callable
import numpy as np

class Neuron:
    def __init__(self, weights_count: int, f_activation_n: int = 1):
        self.weights_count = weights_count
        self.z = None
        self.output = None
        self.f_activation_n = f_activation_n

    @property
    def f_activation(self) -> Callable[[float], float]:
        def relu(z:float) -> float:
            return max(0, z)
        if self.f_activation_n == 1:
            return relu

    @property
    def f_activation_prim(self) -> Callable[[float], float]:
        def relu_grad_mask_prim(z:float) -> float:
                return (z > 0).astype(float)
        if self.f_activation_n == 1:
            return relu_grad_mask_prim

    def init(self) -> Neuron:
        self.wb = np.concatenate([np.full(self.weights_count, 0.01), [0.0]])
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
    def __init__(self, inputs_count: int, neurons_count: int):
        self.inputs_count = inputs_count
        self.neurons_count = neurons_count
        self.neurons = [Neuron(weights_count=inputs_count).init() for _ in range(0, self.neurons_count)]
        self.output = None

    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.output = [self.neurons[n].forward(input_data) for n in range(0, self.neurons_count)]
        return self.output

    @property
    def prims(self)-> np.ndarray:
        return [neuron.prim for neuron in self.neurons]


class NeuralNetwork:
    def __init__(self, inputs_count:int):
        self.inputs_count = inputs_count
        self.output = None
        self.layers: list[Layer] = []

    def addLayer(self, neurons_count: int):
        if len(self.layers) == 0:
            self.layers.append(Layer(self.inputs_count, neurons_count))
            return
        self.layers.append(Layer(self.layers[-1].neurons_count, neurons_count))
        
    def forward(self, input_data:np.ndarray) -> np.ndarray:
        output = input_data;
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return self.output

    def backward(self, train_data:np.ndarray, learning_rate = 0.01):
        layerNum = -1
        layer = self.layers[layerNum]
        if layerNum == -1:
            delta = (self.output - train_data) * layer.prims
        gradients = np.outer(delta, np.concatenate([self.layers[layerNum -1].output, [1]]))
        self.updateWb(layer, gradients, learning_rate)

    def updateWb(self, layer: Layer, gradients,  learning_rate):
        for i, neuron in enumerate(layer.neurons):
            neuron.wb =  neuron.wb - learning_rate * gradients[i]


nn = NeuralNetwork(2)
nn.addLayer(3)
nn.addLayer(2)
output = nn.forward(np.array([1, 2]))

nn.backward(np.array([3, 3]))