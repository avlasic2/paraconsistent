import numpy as np
from FuzzyCMeans import FuzzyCMeans

r1 = np.random.pareto( 3. , (500,10) )
r2 = np.random.normal(-2., 1., (500,10) )

data = np.vstack( (r1, r2) )

model = FuzzyCMeans(data,2)

model.train(power = 2.1, epsilon = .001, runs = 1000)

#select 5 data points from each distribution
indices = [0,1,2,3,4,500,501,502,503,504]

#display data
print( data[ indices ] )

#display the soft predicted group each data point was calculated to be with
print( model.softPredict( indices ) )
