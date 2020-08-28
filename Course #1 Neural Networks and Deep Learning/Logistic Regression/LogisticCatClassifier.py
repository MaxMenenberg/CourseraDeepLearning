import numpy as np
import os
import matplotlib.pyplot as plt
import h5py

# Load in the training data from the .h5 file and extract the 
# training input images, and labels.
def loadTrainingData():
    # Get the training data
    dirName = os.path.dirname(__file__)
    dataSetPath = r'Data Sets\Cats\train_catvnoncat.h5'
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
    dataSetPath = r'Data Sets\Cats\test_catvnoncat.h5'
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
def processTrainingImages(X):
    sampleNum = len(X)
    featureNum = np.product(X[0].shape)
    Xflat = np.zeros((featureNum, sampleNum))
    for n in range(sampleNum):
        Xflat[:,n,None] = X[n].reshape(featureNum,1)/255
    return Xflat

#Sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))    

# Train a logistic regression model
# X: Input data in the form of an (fn x M) matrix where fn is the number of 
# features and M is the number of training samples
# Y: The output labels in the form of an (M x 1) vector
# alpha: Learning rate
# N: Number of training passes over all of the data
def trainLogisticModel(X, Y, N, alpha):
    #Initialize the logistic regression model vectors
    w = np.zeros((Xflat.shape[0],1))
    B = np.zeros((1,Xflat.shape[1]))
    M = X.shape[1]
    
    #Apply gradient descent to the logistic regression model
    for epoch in range(N):
        Z = np.dot(w.T,Xflat) + B
        Yhat = sigmoid(Z)
        dLdw = np.dot((Yhat - Y.T), Xflat.T)/M
        dLdB = np.sum((Yhat - Y.T))/M
        w = w - (alpha*dLdw).T
        B = B - alpha*dLdB    
    
    return {'w':w, 'b':B}

def classifySingleImage(image, model):
    pixelNum = np.product(image.shape)
    x = image.reshape(pixelNum,1)/255
    w = model.get('w')
    B = model.get('b')
    label = sigmoid(np.sum(w*x) + B[0,0])
    if label > 0.5:
        #title = 'This is a cat'
        return 1
    else:
        #title = 'This is not a cat'
        return 0
    #fig, ax = plt.subplots(1,1)
    #ax.imshow(image)
    #ax.set_title(title)
    #plt.tight_layout()
    #plt.show()

plt.close('all')

#Load the data and extract the inputs and outputs
trainingData = loadTrainingData();
Xflat = processTrainingImages(trainingData.get('trainX'))
Y = np.array(trainingData.get('trainY')).reshape(Xflat.shape[1],1)

model = trainLogisticModel(Xflat, Y, 1000, 0.005)



#Test classification accuracy on the training data
trainingImages = trainingData.get('trainX')
trainingN = Xflat.shape[1]
correctLabels = 0
for n in range(trainingN):
    image = trainingImages[n]
    label = classifySingleImage(image, model)
    if label - Y[n] == 0:
        correctLabels += 1
        
print('Classification accuracy on training set: {:.2f}%'.format(correctLabels*100/trainingN) )     

#Test classification accuray on the test data  
testData = loadTestData();
testImages = testData.get('testX')
testY = testData.get('testY')

testN = len(testImages)
correctLabels = 0
for n in range(testN):
    image = trainingImages[n]
    label = classifySingleImage(image, model)
    if label - Y[n] == 0:
        correctLabels += 1
        
print('Classification accuracy on test set: {:.2f}%'.format(correctLabels*100/testN) )                      
                     