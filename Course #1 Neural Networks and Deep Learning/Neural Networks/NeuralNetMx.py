import numpy as np
import math

#Class that describes a layer of a basic neural network
class Layer:
    
    #List of available activation functions to choose from
    actFunList = ['sigmoid', 'tanh', 'relu']
    
    # A layer is described by how many neurons it has and what activation
    # function the neurons uses. The layer will also contain a matrix of
    # weights and a bias but those will be assigned later by the 
    # neural network
    def __init__(self, neuronCount, activationFunction):
        self.neuronCount = neuronCount
        self.activationFunction = activationFunction
        self.weights = np.zeros((1,1))
        self.bias = 0;
        assert activationFunction in self.actFunList, \
            'Activation function not valid. Select a valid activation function'\
                ' [sigmoid, tanh, relu]'
        
    # The layer's activation functions and respective derivatives    
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_dx(x):
        return np.exp(-x)/(np.power((1+np.exp(-x)),2))
    
    def tanh(x):
        return (np.exp(2*x)-1)/(np.exp(2*x)+1)
    
    def tanh_dx(x):
        return 4/(np.power((np.exp(-x)+np.exp(x)),2))
    
    def relu(x):
        y = x
        y[y < 0] = 0
        return y
    
    def relu_dx(x):
        y = x
        y[y <= 0] = 0
        y[y > 0] = 1
        return y
    
    # The dictionary of function pointers that allow the activationFunction 
    # variable to control which function gets called
    __activationFunctionList = {
        'sigmoid':sigmoid,
        'tanh':tanh,
        'relu':relu
        }
    
    __activationFunctionDerivativeList = {
        'sigmoid':sigmoid_dx,
        'tanh':tanh_dx,
        'relu':relu_dx
        }
    
    def actFun(self, x):
        return self.__activationFunctionList[self.activationFunction](x)
    
    def actFunDx(self, x):
        return self.__activationFunctionDerivativeList[self.activationFunction](x)
    
    # A way for the network to assign and update weights and bias
    # for the layer
    def assignWeights(self,w):
        self.weights = w
        
    def assignBias(self,b):
        self.bias = b
        
# A class that describes a basic neural network
class NeuralNetwork:
    
    # A network consists of a list of layers described by their weights and
    # dimensions
    layers = []
    
    # These are terms used for the keeping track of variables for the adaptive 
    # moment estimation optimization i.e. AdamOpt
    adamOpt = False;


    
    CostFunList = ['MSE', 'CrossEntropy']
    def __init__(self, costFunction, learningRate, initialWeightScale, L2Reg = 0):
        self.L2RegLambda = L2Reg;
        self.costFunction = costFunction
        self.learningRate = learningRate
        self.initialWeightScale = initialWeightScale;
        assert costFunction in self.CostFunList, \
            'Activation function not valid. Select a valid activation function'\
                ' [MSE, CrossEntropy]'
    
    def MSELoss(yhat, y):
        return np.power((yhat-y),2)
    
    def MSELossDx(yhat, y):
        return 2*(yhat - y)
    
    def CrossEntroyLoss(yhat, y):
        if((yhat <= 0 ).any()):
            yhat = 0.1; #Avoid negative logs
        elif ((1-yhat <= 0 ).any()):
            yhat = 0.9;
        return -y*np.log(yhat) - (1-y)*np.log(1-yhat)
        
    def CrossEntroyLossDx(yhat, y):
        if((yhat == 0 ).any()):
             yhat = 0.1 #Avoid Divide by 0
        elif ((yhat == 1).any()):
            yhat = 0.9
        return -(y/yhat) + (1-y)/(1-yhat)
        
    
    # The dictionary of function pointers that allow the costFunction 
    # variable to control which function gets called
    __CostFunctionList = {
        'MSE': MSELoss,
        'CrossEntropy':CrossEntroyLoss,
        }
    
    __CostFunctionDerivativeList = {
        'MSE': MSELossDx,
        'CrossEntropy':CrossEntroyLossDx,
        }
    
    def lossFun(self, yhat, y):
        return self.__CostFunctionList[self.costFunction](yhat, y)
    
    def lossFunDx(self, yhat, y):
        return self.__CostFunctionDerivativeList[self.costFunction](yhat,y)
    
    def addLayer(self, layerToAdd):
        self.layers.append(layerToAdd)
     
    # X is input data of the form [nx, m] when nx is the number of features
    # and m is the number of samples
    # Y is of a similar for as X with size [ny, m]
    def initializeNetworkWeights(self, X, Y):
        weightScaling = self.initialWeightScale
        for n in range(len(self.layers)):
            self.layers[n].assignBias(0)
            if(n == 0):
                self.layers[n].assignWeights(weightScaling*np.random.randn(self.layers[n].neuronCount, X.shape[0]))
            else:
                self.layers[n].assignWeights(weightScaling*np.random.randn(self.layers[n].neuronCount, self.layers[n-1].neuronCount))
    
    def getLayers(self):
        return self.layers
                
    def forwardProp(self, X):
        aY = X
        for j in range(len(self.layers)):
            Z = np.matmul(self.layers[j].weights, aY) + self.layers[j].bias
            aY = self.layers[j].actFun(Z)
        return aY;
    
    def backwardProp(self, X, Y, t):
        
        # First we need to do forware propagation to record all of the inputs 
        # and outputs to the layers
        m = X.shape[1]
        aYList = [];
        ZList = [];
        aY = X
        aYList.append(X)
        for j in range(len(self.layers)):
            Z = np.matmul(self.layers[j].weights, aY) + self.layers[j].bias
            ZList.append(Z)
            aY = self.layers[j].actFun(Z)
            aYList.append(aY)
        yhat = aY
        

        # Now we can do back propagation
        dCdyhat = self.lossFunDx(yhat, Y)/m
        delK = np.zeros((1,1))
        for j in range(len(self.layers)-1, -1, -1):
            if (j == len(self.layers)-1):
                d1 = self.layers[j].actFunDx(ZList[j])
                delK = dCdyhat*d1
                dW = np.matmul(delK, aYList[j].T) +  \
                    self.L2RegLambda*self.layers[j].weights/m
                db = np.sum(delK, axis = 1, keepdims=True)
            else:
                d1 = self.layers[j+1].weights
                d2 = self.layers[j].actFunDx(ZList[j])
                delK = np.matmul(d1.T, delK)*d2
                dW = np.matmul(delK, aYList[j].T) + \
                    self.L2RegLambda*self.layers[j].weights/m
                db = np.sum(delK, axis = 1, keepdims=True)
            
            W = self.layers[j].weights
            Bias = self.layers[j].bias
            
            self.layers[j].assignWeights(W - self.learningRate*dW)
            self.layers[j].assignBias(Bias - self.learningRate*db)
            
        Cost = np.sum(self.lossFun(yhat, Y))/m
        return Cost
    
    def train(self,X,Y, iterationNum):
        costList = []
        for n in range(iterationNum):
            cost = self.backwardProp(X,Y,n)
            costList.append(cost)
        return costList
        
        
        