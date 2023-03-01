import numpy as np
from keras.datasets import minst
from keras.utils import np_utils

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict