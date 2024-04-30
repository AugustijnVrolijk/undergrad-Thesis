import xml.etree.ElementTree as ET
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd 
import math

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
    def __init__(self, graph:Graph, iterCount:int, popSize:int, probDecay:float, offset, alpha, beta):
        self.graph = graph
        self.verticeCount = len(graph.distanceMatrix)
        self.correlationMatrix = np.zeros((self.verticeCount, self.verticeCount))
        self.bestMatrix = np.full((self.verticeCount, self.verticeCount), np.inf)
        self.probMatrix = np.empty((self.verticeCount, self.verticeCount))
        self.rng = np.random.default_rng()
        self.mean = 0
        self.solutionCount = 0 
        self.decay = probDecay
        self.offset = offset
        self.alpha = alpha
        self.beta = beta
        self.iterCount = iterCount
        self.popSize = popSize
        self.best = np.inf
        self.bestSol = None

        self.bestSolMatrix = np.empty(iterCount,dtype=float)
        self.meanOverTime = np.empty(iterCount,dtype=float)
        self.sdOverTime = np.empty(iterCount,dtype=float)

        

    def mainRun(self):
        for i in range(self.iterCount):
            self.offsetProbMatrix()
            solutions = np.empty((self.popSize, 2),dtype=object)
            for j in range(self.popSize):
                temp = self.genProbChromosome()
                solutions[j] = temp, self.graph.calcDistance(temp)
            self.batchCalc(solutions)
            print(self.best)
            print(self.bestSol)
            print(i)
            self.bestSolMatrix[i] = self.best
            self.meanOverTime[i] = self.mean
            self.sdOverTime[i] = self.sd

        print("mean is: {}".format(self.mean))
        return self.best
    
    def batchCalc(self, solutions):
        tempMean = 0
        self.variance = 0
        self.meanAbsoluteDeviation = 0
        length = len(solutions)

        for i in range(length):
            sol = solutions[i, 1]
            if sol < self.best:
                self.best = sol
                self.bestSol = solutions[i, 0]
            temp = (sol - self.mean)
            self.variance += temp*temp
            self.meanAbsoluteDeviation += abs(temp)/length
            tempMean += sol/length
        self.variance = self.variance/length
        self.mean = tempMean
        self.sd = math.sqrt(self.variance)

        temp = np.argsort(solutions[:, 1])
        for i in range(length): 
            #reward = self.calcQlearningReward(sol)
            self.testAdd(solutions[i, 1], solutions[i, 0], temp[i])
        return

    def testAdd(self, score, chromosome, rank):
        #normalisedVal = self.calcGaussianReward(score)
        #normalisedVal = self.calcACOReward(rank)
        self.updateQLearningCorrelation(chromosome, score)
        #self.updateCorrelation(chromosome, normalisedVal)
        i = chromosome[-1]
        for f in range(len(chromosome)):
            j = chromosome[f]
            
            if score < self.bestMatrix[i, j] :
                self.bestMatrix[i, j] = score
                self.bestMatrix[j, i] = score
            i = j

    def updateQLearningCorrelation(self, chromosome, score):
        #solutionID format: [(vertexID, edgeID),...,...]
        city1 = chromosome[-1]
        for city2 in chromosome:
            reward = self.calcQlearningReward(score, city1, city2)
            self.correlationMatrix[city1, city2] += reward
            self.correlationMatrix[city2, city1] += reward
            city1 = city2

    def calcQlearningReward(self, score:float, city1, city2):
        lr = 0.2 #learning rate
        dr = 0.4 #discount rate
        maxVal = np.max(self.correlationMatrix[city2])
        reward =  lr*(-score + dr*(maxVal)- self.correlationMatrix[city1, city2])     
        return reward

    def offsetProbMatrix(self):
        minVals = np.min(self.correlationMatrix, axis=1)
        minVal = np.min(minVals)
        """ if minVal == 0:
            self.correlationMatrix += 0.1 
        elif minVal < 0:
            self.correlationMatrix += self.offset*abs(minVal)     
        self.correlationMatrix *= self.decay """

        for i in range(len(self.correlationMatrix)):
            total = 0
            for j in range(len(self.correlationMatrix)):
                cor = ((self.correlationMatrix[i, j]+minVal)**self.alpha)
                bes = (self.bestMatrix[i, j]**self.beta)
                if math.isnan(bes):
                    bes = 1
                elif math.isinf(bes):
                    bes = 1
                if math.isnan(cor):
                    cor = 1
                temp = bes*cor
                self.probMatrix[i, j] = temp
                total += temp
                if total == 0:
                    total = 1
            self.probMatrix[i] = np.divide(self.probMatrix[i], total)        
        return
    
    def calcACOReward(self, rank):
        val = 1/(rank+1)
        return val
    
    def calcGaussianReward(self, score:float):
        p = 0.7
        a = 1
        temp = (math.e)**(-(((score-self.mean)**2)/(2*(self.sd)**2))**p)
        val = a - a*temp
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

    def test(self):
        cur = np.random.randint(0, self.verticeCount)
        solution = []
        solution.append(cur)
        solutionID = [] #vertexID, edgeID
        count = 1
        while count < self.verticeCount:
            edges, edgeID = self.vertices[cur].chooseTransition(solution)
            for edge in edges:
                solution.append(edge)
                count += 1
            solutionID.append((cur, edgeID))
            cur = edges[-1]
       
        return solution, solutionID

    def genRandomChromosome(self):
        cur = np.random.randint(0, self.verticeCount)
        valid = [i for i in range(self.verticeCount)]
        valid.remove(cur)
        solution = []
        solution.append(cur)
        for i in range(self.verticeCount-1):
            cur = np.random.choice(valid)       
            solution.append(cur)
            valid.remove(cur)
        if len(solution) != len(self.correlationMatrix):
            print("sol not correct length")
            exit()
        return solution

    def genProbChromosome(self):
        cur = np.random.randint(0, self.verticeCount)
        valid = [i for i in range(self.verticeCount)]
        valid.remove(cur)
        solution = []
        solution.append(cur)
        for i in range(self.verticeCount-1):
            total = 0
            for j in valid:
                total += self.probMatrix[i, j]
            try:
                val = np.random.uniform(0, total)
            except:
                print(total)
                print(self.probMatrix[i])
                exit()           
            total = 0
            for j in valid:
                total += self.probMatrix[i, j]
                #>= equal needed incase probability is 0 for all transitions
                if total >= val:
                    cur = j
                    break
            solution.append(cur)
            valid.remove(cur)
        if len(solution) != len(self.correlationMatrix):
            print("sol not correct length")
            exit()
        return solution
    
    def normaliseCorr(self):
        minVals = np.min(self.correlationMatrix)
        self.correlationMatrix[self.correlationMatrix >= 0] = -(np.inf)
        maxVals = np.max(self.correlationMatrix)
        min = np.min(minVals)
        max = np.max(maxVals)
        difference = max - min
        for i in range(len(self.correlationMatrix)):
            for j in range(len(self.correlationMatrix)):
                if i == j:
                    self.correlationMatrix[i, j] = None
                    continue
                self.correlationMatrix[i, j] = (self.correlationMatrix[i, j] - min)/difference
        return self.correlationMatrix

    def normaliseDistance(self):
        temp = np.empty((len(self.graph.distanceMatrix),len(self.graph.distanceMatrix)))
        minVals = np.min(self.graph.distanceMatrix)
        min = np.min(minVals)
        maxVals = np.max(self.graph.distanceMatrix)
        max = np.max(maxVals)
        difference = max - min
        for i in range(len(self.graph.distanceMatrix)):
            for j in range(len(self.graph.distanceMatrix)):
                if i == j:
                    temp[i, j] = None
                    continue
                temp[i, j] = 1-((self.graph.distanceMatrix[i, j] - min)/difference)
        return temp

    def normaliseBest(self):
        minVals = np.min(self.bestMatrix)
        min = np.min(minVals)
        self.bestMatrix[self.bestMatrix == np.inf] = 0        
        maxVals = np.max(self.bestMatrix)
        max = np.max(maxVals)
        self.bestMatrix[self.bestMatrix == 0] = None 

        difference = max - min
        for i in range(len(self.bestMatrix)):
            for j in range(len(self.bestMatrix)):
                if i == j:
                    self.bestMatrix[i, j] = None
                    continue
                self.bestMatrix[i, j] = 1-((self.bestMatrix[i, j] - min)/difference)
        return self.bestMatrix
    
def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)


def main():

    graph = fetchGraph('brazil58.xml')
    agent = tester(graph, 700, 500, 0.9, 1.1, 0.8, 0.2)
    best = agent.mainRun()
    print(best)
    
    return

if __name__ == "__main__":
    main()