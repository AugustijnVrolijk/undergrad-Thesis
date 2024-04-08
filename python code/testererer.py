
import numpy as np
import time
#[100,200,200,200,200,200,200,200]
#[132,346,241,753,93,463,321,653,458]
normalisedVals = np.array([1,1,1,1,2,3,4,5,100], dtype=float)
StandardisedVals = np.array([1,1,1,1,2,3,4,5,100], dtype=float)

mean = np.sum(normalisedVals)/len(normalisedVals)
minVal = np.min(normalisedVals)
difference = np.max(normalisedVals) - np.min(normalisedVals)
sd = np.std(normalisedVals)

normalise2 = lambda x: (x-mean)/difference

normalise = lambda x: (x-minVal)/difference
standardise = lambda x: (x-mean)/sd
StandardisedVals = standardise(StandardisedVals)
for i in range(len(normalisedVals)):
    print(normalise2(normalisedVals[i]))
    normalisedVals[i] = normalise(normalisedVals[i])
    

print(normalisedVals)

print(StandardisedVals)