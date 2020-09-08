import numpy as np
import NeuralNetMx as nn
import matplotlib.pyplot as plt


X = np.array([[0,0], 
              [0,1],
              [1,0],
              [1,1]]).T
Y = np.array([[0],
              [1],
              [1],
              [0]]).T


l1 = nn.Layer(3, 'tanh')
l2 = nn.Layer(2, 'tanh')
l3 = nn.Layer(1, 'sigmoid')

Net = nn.NeuralNetwork('MSE', 1.5, 0.1)
Net.addLayer(l1)
Net.addLayer(l2)
Net.addLayer(l3)

Net.initializeNetworkWeights(X,Y)
print(Net.layers[2].weights)
L = 10000
cost = Net.train(X,Y,L)
print(Net.layers[2].weights)
t = np.arange(1,L+1,1)

fig, ax = plt.subplots()
ax.plot(t,cost)
ax.ticklabel_format(useOffset=False)
plt.show()
yhat = Net.forwardProp(X)
print(np.round(yhat))

