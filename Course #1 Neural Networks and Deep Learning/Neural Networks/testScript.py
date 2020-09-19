import numpy as np
import NeuralNetMx as nn
import matplotlib.pyplot as plt


def relu(x):
    if (x.all() < 0):
        return 0
    else:
        return x.all()

x = np.array([[1,2,3,4],
              [-1,-2,-3,-4],
              [0, 5, -6, 10]])

y = x
y[y<=0]=0