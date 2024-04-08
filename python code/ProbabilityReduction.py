
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
        self.countMatrix = np.zeros((self.verticeCount, self.verticeCount))
        self.probMatrix = np.empty((self.verticeCount, self.verticeCount))
        self.minLenMatrix = np.full((self.verticeCount, self.verticeCount), np.inf)
        self.rng = np.random.default_rng()
        self.mean = 0
        self.decay = probDecay
        self.iterCount = iterCount
        self.popSize = popSize
        self.oversixCount = 0
        self.graphdata = np.zeros((iterCount, 4))

    def mainRun(self):
        for i in range(self.iterCount):
            self.offsetProbMatrix()
            solutions = np.empty((self.popSize, 2),dtype=object)
            for j in range(self.popSize):
                temp = self.genProbChromosome()
                solutions[j] = temp, self.graph.calcDistance(temp)
            self.batchCalc(solutions[:, 1])

            for j in solutions:
                if j[1] < 4200:
                    self.graphdata[i, 0] = self.graphdata[i, 0] + 1
                    if j[1] < 3900:
                        self.graphdata[i, 1] = self.graphdata[i, 1] + 1
                        if j[1] < 3600:
                            self.graphdata[i, 2] = self.graphdata[i, 2] + 1
                            if j[1] < 3400:
                                print(j[1])
                                self.graphdata[i, 3] = self.graphdata[i, 3] + 1

                self.updateCountMatrix(j[0])
                normalisedVal = self.getNormalisedVal(j[1])
                self.updateCorrelation(j[0], normalisedVal)
                self.checkIfMin(j[0], j[1])
            self.genBestGuess()
            print(i)
            
        self.genBestGuess()
        return
    
    def updateCountMatrix(self, solution):
        i = solution[-1]
        for f in range(len(solution)):
            j = solution[f]
            self.countMatrix[i][j] += 1
            i = j
    
    def checkIfMin(self, solution, score):
        avLen = score/self.verticeCount
        parent1 = solution[-1]
        for parent2 in solution:
            if avLen < self.minLenMatrix[parent1][parent2]:
                self.minLenMatrix[parent1][parent2] = avLen
                self.minLenMatrix[parent2][parent1] = avLen
            parent1 = parent2
        return

    def batchCalc(self, solutions):
        tempMean = 0
        self.variance = 0
        self.meanAbsoluteDeviation = 0
        length = len(solutions)

        for sol in solutions:
            temp = (sol - self.mean)

            self.variance += temp*temp
            self.meanAbsoluteDeviation += abs(temp)/length
            tempMean += sol/length
        
        self.variance = self.variance/length
        self.mean = tempMean
        self.sd = math.sqrt(self.variance)
        return
    
    def offsetProbMatrix(self):
        minVals = np.min(self.correlationMatrix, axis=1)
        for i in range(len(self.correlationMatrix)):
            self.probMatrix[i] = self.correlationMatrix[i] + 1.01*abs(minVals[i])
            total = np.sum(self.probMatrix[i])
            self.probMatrix[i] = np.divide(self.probMatrix[i], total)
        
        self.correlationMatrix *= self.decay
        return

    def getNormalisedVal(self, score:float):
        p = 1
        a = 10
        temp = (math.e)**(-(((score-self.mean)**2)/(2*(self.sd)**2))**p)
        val = a - a*temp
        if val == 6:
            self.oversixCount += 1

        if score > self.mean:
            val = -abs(val)
        return val

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

        curCity = np.random.choice(validGuesses)
        validGuesses.remove(curCity)
        bestGuess = [curCity]
        
        for i in range(self.verticeCount-1):
            temp = -10000000
            chosenCity = None
            for city in validGuesses:
                if self.correlationMatrix[curCity][city] > temp:
                    temp = self.correlationMatrix[curCity][city]
                    chosenCity = city
            bestGuess.append(chosenCity)
            validGuesses.remove(chosenCity)

        print("best guess : {} has value {} \n".format(bestGuess, self.graph.calcDistance(bestGuess)))
        print("mean: {}  sd: {} \n".format(self.mean, self.sd))
        print("MAD: {}".format(self.meanAbsoluteDeviation))
        return

    def genProbChromosome(self):
        cur = np.random.randint(0, self.verticeCount)
        valid = [i for i in range(self.verticeCount)]
        valid.remove(cur)
        solution = []
        solution.append(cur)
        for i in range(self.verticeCount-1):
            prob = np.delete(self.probMatrix[cur], solution)
            constant = 1/np.sum(prob)
            prob *= constant
            cur = self.rng.choice(valid,p=prob)
            solution.append(cur)
            valid.remove(cur)
           
        return solution

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)


def main():
    
    graph = fetchGraph('burma14.xml')
    agent = tester(graph, 100, 200, 0.9)
    agent.mainRun()
    print("over 6")
    print(agent.oversixCount)
    # convert array into dataframe 
    DF = pd.DataFrame(agent.graphdata) 
    DF2 = pd.DataFrame(graph.distanceMatrix)
    DF3 = pd.DataFrame(agent.correlationMatrix)
    DF4 = pd.DataFrame(agent.countMatrix)
    # save the dataframe as a csv file 
    DF.to_csv("data1.csv")
    DF2.to_csv("data2.csv")
    DF3.to_csv("correlationMatrix.csv")
    DF4.to_csv("countMatrix.csv")
    return

if __name__ == "__main__":
    main()