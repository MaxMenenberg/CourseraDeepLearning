import numpy as np
import NeuralNetMx as nn
import planar_utils
import matplotlib.pyplot as plt
import os
import h5py


plt.close('all')

# Load in the training data from the .h5 file and extract the 
# training input images, and labels.
def loadTrainingData():
    # Get the training data
    dirName = os.path.dirname(__file__)
    dataSetPath = r'DataSets\train_catvnoncat.h5'
    fileName = os.path.join(dirName, dataSetPath)

    with h5py.File(fileName, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        classKey = list(f.keys())[0]
        trainxKey = list(f.keys())[1]
        trainyKey = list(f.keys())[2]
    
        # Get the data
        classData = list(f[classKey])
        trainSetX = list(f[trainxKey])
        trainSetY = list(f[trainyKey])
    return {'ClassLabels':classData, 'trainX':trainSetX, 'trainY':trainSetY}

def loadTestData():
    # Get the test data
    dirName = os.path.dirname(__file__)
    dataSetPath = r'DataSets\test_catvnoncat.h5'
    fileName = os.path.join(dirName, dataSetPath)

    with h5py.File(fileName, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        classKey = list(f.keys())[0]
        testxKey = list(f.keys())[1]
        testyKey = list(f.keys())[2]
    
        # Get the data
        classData = list(f[classKey])
        testSetX = list(f[testxKey])
        testSetY = list(f[testyKey])
    return {'ClassLabels':classData, 'testX':testSetX, 'testY':testSetY}

# Flatten the training images and normalize their RGB values to [0,1]
def flattenImageData(X):
    sampleNum = len(X)
    featureNum = np.product(X[0].shape)
    Xflat = np.zeros((featureNum, sampleNum))
    for n in range(sampleNum):
        Xflat[:,n,None] = X[n].reshape(featureNum,1)/255
    return Xflat

#Load the training data and extract the inputs and outputs
trainingData = loadTrainingData();
XflatTrain = flattenImageData(trainingData.get('trainX'))
Ytrain = np.array(trainingData.get('trainY')).reshape(1, XflatTrain.shape[1])

#Load the training data and extract the inputs and output
testData = loadTestData()
XflatTest = flattenImageData(testData.get('testX'))
Ytest = np.array(testData.get('testY')).reshape(1, XflatTest.shape[1])


Net = nn.NeuralNetwork('MSE', 0.125, 0.05, 0.05)


l1 = nn.Layer(60, 'tanh')
l2 = nn.Layer(14, 'tanh')
l3 = nn.Layer(5, 'tanh')
l5 = nn.Layer(1, 'sigmoid')

Net.addLayer(l1)
Net.addLayer(l2)
Net.addLayer(l3)
Net.addLayer(l5)

Net.initializeNetworkWeights(XflatTrain,Ytrain)
epoch = 2000
print(Net.layers[0].weights)
cost = Net.train(XflatTrain,Ytrain,epoch)
print(Net.layers[0].weights)
t = np.arange(1,epoch+1,1)

fig, ax = plt.subplots()
ax.plot(t,cost)
ax.ticklabel_format(useOffset=False)
plt.show()
yhatTrain = np.round(Net.forwardProp(XflatTrain))

correctPredictions = 0;
Mtrain = Ytrain.shape[1]
for n in range(Mtrain):
    if(Ytrain[0,n] == yhatTrain[0,n]):
        correctPredictions = correctPredictions + 1;

print('Classification accuracy on training set: {:.2f}%'.format(correctPredictions*100/Mtrain) )

yhatTest = np.round(Net.forwardProp(XflatTest))
correctPredictions = 0
Mtest = Ytest.shape[1]
for n in range(Mtest):
    if(Ytest[0,n] == yhatTest[0,n]):
        correctPredictions = correctPredictions + 1;

print('Classification accuracy on test set: {:.2f}%'.format(correctPredictions*100/Mtest) )  

