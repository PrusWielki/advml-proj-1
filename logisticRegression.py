import numpy as np


class LogisticRegression:
    def __init__(self, noOfIterations, learningRate):
        self.w= []
        self.noOfIterations = noOfIterations
        self.learningRate = learningRate
        self.costs=[]
        pass
    def fit(self,X,y):
        self.w = np.random.randint(0,1,size=X.shape[1])
        self.bias=0
        self.costs=[]
        for i in range(self.noOfIterations):
            
            a = np.dot(X,self.w) +self.bias

            yHat=1/(1+np.exp(-a))

            # cost function (cross entropy loss)
            self.costs.append((-1/X.shape[0])(np.dot(y,yHat)+np.dot((1-y),np.log(1-yHat))))
            
            # update weights
            weightChange = (1/X.shape[0])*np.dot(X.T,(yHat-y))
            biasChange = (1/X.shape[0]) * np.sum(yHat-y)

            self.w -= self.learningRate*weightChange
            self.bias-=self.learningRate*biasChange

    def predict(self,X):
        predicted  = 1/(1+np.exp(-(np.dot(X,self.w) +self.bias)))
        predictedClasses = []
        for x in predicted:
            if x>0.5:
                predictedClasses.append(1)
            else:
                predictedClasses.append(0)

        return predictedClasses

            


