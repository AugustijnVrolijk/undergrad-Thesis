import numpy as np
import pandas as pd
import math 
import statistics    
import csv

def fetchData(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def cleanAndReturnMinMax(array):

    array.pop(0)
    temp = np.array(array, dtype=float)
    temp = np.delete(temp, 0, axis=1)
    temp[temp == np.inf] = 0
    max = np.max(temp, axis=1)
    temp[temp == 0] = np.inf
    min = np.min(temp, axis=1)

    return min, max, temp

def main():
    data = fetchData('data2.csv')
    minDistance, maxDistance, distanceMatrix = cleanAndReturnMinMax(data)
    maxDistance = maxDistance - minDistance
    difDistanceMatrix = np.empty((len(distanceMatrix[0]), len(distanceMatrix[0])),dtype=float)
    for i in range(len(distanceMatrix)): #row
        for j in range(len(distanceMatrix[0])): #col
            if minDistance[i] < minDistance[j]:
                difDistanceMatrix[i, j] = distanceMatrix[i, j] - minDistance[i]
            else:
                difDistanceMatrix[i, j] = distanceMatrix[i, j] - minDistance[j]
            if maxDistance[i] > maxDistance[j]:
                difDistanceMatrix[i, j] = difDistanceMatrix[i, j]/maxDistance[i]
            else:
                difDistanceMatrix[i, j] = difDistanceMatrix[i, j]/maxDistance[j]


    data = fetchData('data1.csv')
    minLen, maxLen, minLenMatrix = cleanAndReturnMinMax(data)
    maxLen = maxLen - minLen
    difMinLenMatrix = np.empty((len(minLenMatrix[0]), len(minLenMatrix[0])),dtype=float)
    for i in range(len(minLenMatrix)): #row
        for j in range(len(minLenMatrix[0])): #col
            if minLen[i] < minLen[j]:
                difMinLenMatrix[i, j] = minLenMatrix[i, j] - minLen[i]
            else:
                difMinLenMatrix[i, j] = minLenMatrix[i, j] - minLen[j]
            if maxLen[i] > maxLen[j]:
                difMinLenMatrix[i, j] = difMinLenMatrix[i, j]/maxLen[i]
            else:
                difMinLenMatrix[i, j] = difMinLenMatrix[i, j]/maxLen[j]

    print(difMinLenMatrix)
    print(difDistanceMatrix)
    print("\n\n\n")
    print(maxLen)
    print(maxDistance)
    print(minDistance)
    #print(distanceMatrix)
    #print(minLenMatrix)
    DF = pd.DataFrame(difMinLenMatrix)
    # save the dataframe as a csv file 
    DF.to_csv("data3.csv")
    return

if __name__ == "__main__":
	main()