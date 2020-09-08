import numpy as np
import NeuralNetMx as nn
import planar_utils
import matplotlib.pyplot as plt

plt.close('all')
X, Y = planar_utils.load_planar_dataset()

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

Net = nn.NeuralNetwork('MSE', 1.2, 0.1)

l1 = nn.Layer(4, 'tanh')
l2 = nn.Layer(4, 'tanh')
l3 = nn.Layer(4, 'tanh')
l4 = nn.Layer(1, 'sigmoid')

Net.addLayer(l1)
Net.addLayer(l4)

Net.initializeNetworkWeights(X,Y)
epoch = 9999
print(Net.layers[0].weights)
cost = Net.train(X,Y,epoch)
print(Net.layers[0].weights)
t = np.arange(1,epoch+1,1)

fig, ax = plt.subplots()
ax.plot(t,cost)
ax.ticklabel_format(useOffset=False)
plt.show()
yhat = np.round(Net.forwardProp(X))

correctPredictions = 0;
M = Y.shape[1]
for n in range(M):
    if(Y[0,n] == yhat[0,n]):
        correctPredictions = correctPredictions + 1;

print('Classification accuracy on training set: {:.2f}%'.format(correctPredictions*100/M) )  

fig2, ax2 = plt.subplots()
ax2.scatter(X[0, :], X[1, :], c=yhat, s=40, cmap=plt.cm.Spectral);
