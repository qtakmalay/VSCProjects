from activation import Activation
import numpy as np

class Tahn(Activation): 
    def __init__(self):
        tahn = lambda x: np.tanh(x) ** 2
        tahn_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tahn, tahn_prime)

class Sigmoid(Activation):
    def __init__(self, activation, activation_prime):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super.__init__(sigmoid, sigmoid_prime)