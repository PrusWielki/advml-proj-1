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
                self.fitSgd(X,y)
            case Optimizer.Adam:
                self.fitAdam(X,y)
            
    def predict(self,X):
        predicted  = 1/(1+np.exp(-(np.dot(X,self.w).astype(float) +self.bias)).astype(float))
        predictedClasses = []
        for x in predicted:
            if x>0.5:
                predictedClasses.append(1)
            else:
                predictedClasses.append(0)

        return predictedClasses
    
    def fitSgd(self, X,y):
        # Initializing with weights 0 and 0 bias
        self.w = np.zeros(X.shape[1])
        self.bias=0
        self.costs=[]
        for i in range(self.noOfIterations):
            
            a = np.dot(X,self.w) +self.bias
            a=a.astype(float)
            yHat=1/(1+np.exp(-a))

            # cost function (cross entropy loss)
            self.costs.append((-1/X.shape[0])*(np.dot(y,np.log(yHat))+np.dot((1-y),np.log(1-yHat))))
            
            # update weights
            weightChange = (1/X.shape[0])*np.dot(X.T,(yHat-y))
            biasChange = (1/X.shape[0]) * np.sum(yHat-y)

            self.w = self.w+self.learningRate*weightChange
            self.bias=self.bias+self.learningRate*biasChange
    def fitAdam(self,X,y):

        # 1. Init values
        # 2. Compute Gradients
        # 3. Get first and second moments
        # 4. Calculate bias corrections
        # 5. update weights
        self.w = np.zeros(X.shape[1])
        self.bias=0
        self.costs=[]
        # Values used by TensorFlow: beta1=0.9, beta2=0.999, epsilon=1e-08
        beta1=0.9
        beta2=0.999
        epsilon=1e-08
        moment1Weights=np.zeros(X.shape[1])
        moment2Weights=np.zeros(X.shape[1])
        moment1Bias=0
        moment2Bias=0
        mhat=0
        vhat=0
        
        for i in range(self.noOfIterations):
            
            a = np.dot(X,self.w) +self.bias
            a=a.astype(float)
            yHat=1/(1+np.exp(-a))

            # cost function (cross entropy loss)
            self.costs.append((-1/X.shape[0])*(np.dot(y,np.log(yHat))+np.dot((1-y),np.log(1-yHat))))
            
            # Calculate gradient of cross entropy with respect to weights and bias
            # From what I understand the gradient is simply y predicted - y but in this article https://medium.com/analytics-vidhya/implementing-logistic-regression-with-sgd-from-scratch-5e46c1c54c35
            # they claim that gradient with respect to weights should be x(y-y_pred) and with respect to bias: y- ypred
            gradientWithRespectToWeights=np.dot(X.T,yHat-y)
            gradientWithRespectToBias = yHat-y
            moment1Weights=beta1*moment1Weights+(1-beta1)*gradientWithRespectToWeights
            moment2Weights=beta2*moment2Weights+(1-beta2)*gradientWithRespectToWeights*gradientWithRespectToWeights

            mhatWeights = moment1Weights/(1-beta1**i)
            vhatWeights = moment2Weights/(1-beta2**i)

            self.w = self.w-self.learningRate*mhatWeights/(np.sqrt(vhatWeights)+epsilon)

            moment1Bias=beta1*moment1Bias+(1-beta1)*gradientWithRespectToBias
            # We could iterate over all xs and do it one by one, but I think it is equivalent to summing the moment2 for all xs and updating the bias at once
            moment2Bias=np.dot(beta2*moment2Bias+(1-beta2),gradientWithRespectToBias*gradientWithRespectToBias)

            mhatBias = moment1Bias/(1-beta1**i)
            vhatBias = moment2Bias/(1-beta2**i)

            self.bias = self.bias-self.learningRate*mhatBias/(np.sqrt(vhatBias)+epsilon)

    def fitIwls(self,X,y):
        pass           