import numpy as np
import NeuralNetMx as nn
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

dataPath = r'C:\Users\Max\Desktop\My Documents\Coursera Courses\Machine Learning\CourseraDeepLearning\Course #1 Neural Networks and Deep Learning\Logistic Regression\Data Sets\Basket Ball Games'
dataFileName = r'\nba.games.stats.csv';

# Read and parse the data we want to use for classification
nbaData = pd.read_csv(dataPath + dataFileName)
winLoss = (nbaData.iloc[:, 6] == 'W').astype('uint8').to_numpy().reshape(-1,1)
nbaDataStats = nbaData.iloc[:, 12:nbaData.shape[1]-1].to_numpy()
sampleNum = nbaDataStats.shape[0]

# Normalize the stats data
for n in range(nbaDataStats.shape[1]):
    tempData = nbaDataStats[:,n]
    mu = np.mean(tempData)
    sig = np.std(tempData)
    nbaDataStats[:, n] = (tempData - mu)/sig
    
nbaDataNorm = np.concatenate((nbaDataStats, winLoss), axis = 1)

# Since the data is in chronological order we will shuffle the data 
# before splitting into training and testing

shuffles = int(sampleNum/2)
for n in range(shuffles):
    # Pick two random rows and swap them and keep doing this
    rowAIndex = int(np.random.rand(1)*sampleNum)
    rowBIndex = int(np.random.rand(1)*sampleNum)
    
    tempRow = nbaDataNorm[rowAIndex, :]
    nbaDataNorm[rowAIndex, :] = nbaDataNorm[rowBIndex, :]
    nbaDataNorm[rowBIndex, :] = tempRow;
    

trainX = nbaDataNorm[1:8000, 1:nbaDataNorm.shape[1]-2].T
trainY = nbaDataNorm[1:8000, nbaDataNorm.shape[1]-1].T.reshape(1,-1)

testX = nbaDataNorm[8001:nbaDataNorm.shape[0]-1, 1:nbaDataNorm.shape[1]-2].T
testY = nbaDataNorm[8001:nbaDataNorm.shape[0]-1, nbaDataNorm.shape[1]-1].reshape(1,-1)

Net = nn.NeuralNetwork('MSE', 0.5, 0.05)


l1 = nn.Layer(10, 'tanh')
l2 = nn.Layer(5, 'tanh')
l3 = nn.Layer(2, 'tanh')
l4 = nn.Layer(1, 'sigmoid')

Net.addLayer(l1)
Net.addLayer(l2)
Net.addLayer(l3)
Net.addLayer(l4)

Net.initializeNetworkWeights(trainX,trainY)
epoch = 1700
print(Net.layers[0].weights[0,0])
cost = Net.train(trainX,trainY,epoch)
print(Net.layers[0].weights[0,0])
t = np.arange(1,epoch+1,1)

fig, ax = plt.subplots()
ax.plot(t,cost)
ax.ticklabel_format(useOffset=False)
plt.show()
yhatTrain = np.round(Net.forwardProp(trainX))

correctPredictions = 0;
Mtrain = trainY.shape[1]
for n in range(Mtrain):
    if(trainY[0,n] == yhatTrain[0,n]):
        correctPredictions = correctPredictions + 1;

print('Classification accuracy on training set: {:.2f}%'.format(correctPredictions*100/Mtrain) )


yhatTest = np.round(Net.forwardProp(testX))
correctPredictions = 0
Mtest = testY.shape[1]
for n in range(Mtest):
    if(testY[0,n] == yhatTest[0,n]):
        correctPredictions = correctPredictions + 1;

print('Classification accuracy on test set: {:.2f}%'.format(correctPredictions*100/Mtest) ) 
    
    