
import xml.etree.ElementTree as ET
import numpy as np
import math
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd 
import time

class vertex():
    def __init__(self) -> None:
        self.closestVertices = None
        return
    
    def getNClosest(self, distance:int, edges:list) -> None:
        return

class Graph():
    def __init__(self, XMLgraph) -> None:
        self.distanceMatrix = np.zeros((len(XMLgraph),len(XMLgraph)))
        parent1 = 0
        for Vertex in XMLgraph.iter('vertex'):
            for Edge in Vertex.iter('edge'):
                parent2 = int(Edge.text)
                weight = int(float(Edge.get('cost')))
                self.distanceMatrix[parent1][parent2] = weight
            parent1 += 1
                
    def calcDistance(self, circuitVector :list) -> int: 
        
        distance = 0
        i = circuitVector[-1]
        for f in range(len(circuitVector)):
            j = circuitVector[f]
            distance += self.distanceMatrix[i][j]
            i = j

        return distance

class tester():
    def __init__(self, graph:Graph, iterCount:int, popSize:int, probDecay:float):
        self.graph = graph
        self.verticeCount = len(graph.distanceMatrix)
        self.correlationMatrix = np.ones((self.verticeCount, self.verticeCount))
        self.probMatrix = np.empty((self.verticeCount, self.verticeCount))
        self.rng = np.random.default_rng()
        self.mean = 0
        self.solutionCount = 0 
        self.decay = probDecay
        self.iterCount = iterCount
        self.popSize = popSize

    def mainRun(self):
        for i in range(self.iterCount):
            self.offsetProbMatrix()
            solutions = np.empty(100,dtype=object)
            for j in range(self.popSize):
                temp = self.genProbChromosome()
                solutions[j] = (temp, self.graph.calcDistance(temp))
            for j in solutions:
                self.testAdd(j[1], j[0])
            self.genBestGuess()
            print(i)
            
        self.genBestGuess()
        print(self.correlationMatrix)
        print("mean is: {}".format(self.mean))
        return

    def testAdd(self, score, chromosome):
        self.updateMean(score)
        normalisedVal = self.getNormalisedVal(score)
        self.updateCorrelation(chromosome, normalisedVal)

    def offsetProbMatrix(self):
        minVals = np.min(self.correlationMatrix, axis=1)
        for i in range(len(self.correlationMatrix)):
            self.probMatrix[i] = self.correlationMatrix[i] + 1.01*abs(minVals[i])
            total = np.sum(self.probMatrix[i])
            self.probMatrix[i] = np.divide(self.probMatrix[i], total)
        
        self.correlationMatrix *= self.decay
        return

    def getNormalisedVal(self, score):
        self.f = (1*(6**(-18)))
        temp = abs(self.f*((score-5000)**10))
        if temp > 10:
            temp = 10
            print(temp)
            print(score)
        if score > self.mean:
            temp = -abs(temp)
        return temp 
    
    def updateMean(self, newVal):
        self.solutionCount += 1
        temp = newVal - self.mean
        self.mean += temp/self.solutionCount
        return

    def updateCorrelation(self, chromosome, normalisedVal):
        city1 = chromosome[-1]
        for city2 in chromosome:			
            self.correlationMatrix[city1][city2] += normalisedVal
            self.correlationMatrix[city2][city1] += normalisedVal
            city1 = city2
        
    def genBestGuess(self) -> list:
        
        validGuesses = []
        for i in range(self.verticeCount):
            validGuesses.append(i)

        bestGuess = [0]
        curCity = 0
        for i in range(self.verticeCount):
            temp = -10000000
            chosenCity = None
            for city in validGuesses:
                if self.correlationMatrix[curCity][city] > temp:
                    temp = self.correlationMatrix[curCity][city]
                    chosenCity = city
            bestGuess.append(chosenCity)
            validGuesses.remove(chosenCity)

        print("best guess : {} has value {} \n".format(bestGuess, self.graph.calcDistance(bestGuess)))
        return

    def genProbChromosome(self):
        cur = np.random.randint(0, self.verticeCount)
        valid = [i for i in range(self.verticeCount)]
        valid.remove(cur)
        solution = []
        solution.append(cur)
        for i in range(self.verticeCount-1):
            try:
                prob = np.delete(self.probMatrix[cur], solution)
                constant = 1/np.sum(prob)
                prob *= constant
                cur = self.rng.choice(valid,p=prob)
                solution.append(cur)
                valid.remove(cur)
            except:
                print("prob: ",prob)
                print("constant: ", constant)
                print("i: ",i)
                print("solution: ",solution)
                print("valid: ", valid)
                print("probmatrix: ",self.probMatrix[cur])
                print(self.probMatrix.tolist())
                print(self.correlationMatrix.tolist())
                exit()

        return solution

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)


def main():
    
    graph = fetchGraph('burma14.xml')
    length = len(graph.distanceMatrix)
    agent = tester(length)
    
    
    # convert array into dataframe 
    DF = pd.DataFrame(agent.correlationMatrix) 
    DF2 = pd.DataFrame(graph.distanceMatrix)
    # save the dataframe as a csv file 
    DF.to_csv("data1.csv")
    DF2.to_csv("data2.csv")

    return

if __name__ == "__main__":
    main()