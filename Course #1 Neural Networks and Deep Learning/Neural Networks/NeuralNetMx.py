import numpy as np

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
        if (x < 0):
            return 0
        else:
            return x
    
    def relu_dx(x):
        if (x < 0):
            return 0
        else:
            return 1
    
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