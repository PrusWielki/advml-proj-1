import numpy as np
from enum import Enum

class Optimizer(Enum):
    SGD=1
    Adam=2
    IWLS=3
class LogisticRegression:
    def __init__(self, noOfIterations: int=1000, learningRate: float=0.001, optimizer: Optimizer=Optimizer.SGD):
        self.w= []
        self.noOfIterations = noOfIterations
        self.learningRate = learningRate
        self.costs=[]
        self.optimizer = optimizer
        
    def fit(self,X,y):
        match self.optimizer:
            case Optimizer.SGD:
                self.fitSGD(X,y)
            
    def predict(self,X):
        predicted  = 1/(1+np.exp(-(np.dot(X,self.w).astype(float) +self.bias)).astype(float))
        predictedClasses = []
        for x in predicted:
            if x>0.5:
                predictedClasses.append(1)
            else:
                predictedClasses.append(0)

        return predictedClasses
    
    def fitSGD(self, X,y):
        self.w = np.zeros(X.shape[1])
        self.bias=0
        self.costs=[]
        for i in range(self.noOfIterations):
            
            a = np.dot(X,self.w) +self.bias
            a=a.astype(float)
            yHat=1/(1+np.exp(-a))

            # cost function (cross entropy loss)
            self.costs.append((-1/X.shape[0])*(np.dot(y,yHat)+np.dot((1-y),np.log(1-yHat))))
            
            # update weights
            weightChange = (1/X.shape[0])*np.dot(X.T,(yHat-y))
            biasChange = (1/X.shape[0]) * np.sum(yHat-y)

            self.w = self.w+self.learningRate*weightChange
            self.bias=self.bias+self.learningRate*biasChange