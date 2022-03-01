import numpy as np
from typing import Union
from tqdm import tqdm

class FuzzyCMeans():
    def __init__(self, data:np.ndarray, numberClusters:int, useKLDiv:bool = False ):
        self.numberClusters = numberClusters
        self.useKLDiv = useKLDiv

        if useKLDiv:
            self.data = self.maptoSimplex( data )
        else:
            self.data = data

        self.clusterCenters = np.random.normal( 0., 1. , (numberClusters, data.shape[1]) )
        if useKLDiv:
            self.clusterCenters = self.maptoSimplex( self.clusterCenters )

        unnormalizedWeights = np.random.pareto( 5., ( data.shape[0] , numberClusters ) )
        self.weights = unnormalizedWeights/unnormalizedWeights.sum(1).reshape(-1,1)

        if useKLDiv:
            self.dist = self._dataCenterDistKL
        else:
            self.dist = self._dataCenterDistLP


    def maptoSimplex(self, arr:np.ndarray)->np.ndarray:
        expMap = np.hstack( ( np.exp(arr), np.ones( ( arr.shape[0] , 1  ) ) ) )
        return expMap/expMap.sum(1).reshape(-1,1)


    def updateCenters(self, power:float):
        weightsCalc = ( self.weights.copy() )**power
        self.clusterCenters = np.dot( ( weightsCalc / weightsCalc.sum(0) ).T, self.data )


    def updateWeights(self, power:float, lp:float):
        dist = ( self.dist( self.data, self.clusterCenters, lp) )**( 2./ (power-1.) )
        self.weights = (1./( np.einsum('ij,ik->jik', dist, dist**(-1)) ).sum(2) ).T


    def _dataCenterDistLP(self, ar1:np.ndarray, ar2:np.ndarray, lp:float = 2.)->np.ndarray:
        return ( ( ( ar1[:, np.newaxis] - ar2 )**lp ).sum(2) )**(1./lp)


    def _dataCenterDistKL(self, ar1:np.ndarray, ar2:np.ndarray, lp=2.)->np.ndarray:
        return (-1.) * ( np.array([ar2] * ar1.shape[0]) * np.log(ar1[:, np.newaxis] * ar2 ** (-1)) ).sum(2)

    def train(self, power:float=2., lp:float=2.,  epsilon:float=.01, runs:float = 1000):
        previousWeights = self.weights.copy()

        err = 2*epsilon
        pbar = tqdm(desc='Error', total=runs + 1)
        pbar.set_postfix(Train_loss=err)
        iter = 0

        while (err > epsilon) and ( iter < runs+1 ):
            self.updateWeights(power=power, lp=lp)
            self.updateCenters(power=power)
            err = np.abs( previousWeights - self.weights ).max()
            previousWeights = self.weights.copy()
            iter += 1
            pbar.update(1)
            pbar.set_postfix(Train_loss=err)
        pbar.close()


    def softPredict(self,Indices:Union[np.ndarray, list]=None)->np.ndarray:
        if Indices == None:
            return self.weights.argmax(1)
        return self.weights[ Indices ].argmax(1)

