import numpy as np


class LogisticRegression:
    def __init__(self, noOfIterations):
        self.w= []
        self.noOfIterations = noOfIterations
        pass
    def fit(self,X):
        self.w = np.random.randint(0,1,size=X.shape[1])
        costs=[]
        for i in range(self.noOfIterations):
            
            a = np.dot(X,self.w)
            # link function
            yHat=1/(1+np.exp(-a))

            # cost function
            cost = (-1/X.shape[0])


