import numpy as np
from tqdm import tqdm

class FuzzyCMeans():
    def __init__(self,data,numberClusters):
        self.data = data
        self.numberClusters = numberClusters
        self.clusterCenters = np.random.normal( 0., 1. , (numberClusters, data.shape[1]) )
        self.weights = np.random.weibull( 5., ( data.shape[0] , numberClusters ) )

    def updateCenters(self, power):
        weightsCalc = ( self.weights.copy() )**power
        self.clusterCenters = np.dot( ( weightsCalc / weightsCalc.sum(0) ).T, self.data )

    def updateWeights(self, power):
        EucDist = ( self._dataCenterDist( self.data, self.clusterCenters) )**( 2./ (power-1.) )
        self.weights = (1./( np.einsum('ij,ik->jik', EucDist, EucDist**(-1)) ).sum(2) ).T

    def _dataCenterDist(self, ar1, ar2, lp = 2.):
        return ( ( ( ar1[:, np.newaxis] - ar2 )**lp ).sum(2) )**(1./lp)

    def train(self, power = 2., epsilon = .01, runs = 1000):
        previousWeights = self.weights.copy()

        err = 2*epsilon
        pbar = tqdm(desc='Error', total=runs + 1)
        pbar.set_postfix(Train_loss=err)
        iter = 0

        while (err > epsilon) and ( iter < runs+1 ):
            self.updateWeights(power=power)
            self.updateCenters(power=power)
            err = np.abs( previousWeights - self.weights ).max()
            previousWeights = self.weights.copy()
            iter += 1
            pbar.update(1)
            pbar.set_postfix(Train_loss=err)
        pbar.close()

    def softPredict(self,Indices=None):
        if Indices == None:
            return self.weights.argmax(1)
        return self.weights[ Indices ].argmax(1)