
import xml.etree.ElementTree as ET
import numpy as np
import math
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd 
import time

"""




















POSSIBLE BUG:::::::::::::
    when generating a solution:
        a vertice called may have deleted transition containing the allowed next vertices..
        i.e. solution for graph 1->5 is: 1->3->5->2
        but vertice 2 removed transition 2-> 4 as a possible move.

        In this case either reroll, or lower probability of transitions making partial solution
        (1->3, 3->5 and 5->2)


possible bug,
    when analysing a solution:
        removing edge, then in future iterations if that edge has been removed the index may fail for changing correlation etc..




































"""

class vertex():
    def __init__(self, edges, decay) -> None:

        self.nEdges = len(edges)
        self.edges = edges #list of lists, each sub list represents the vertices it goes through in the transition
        self.edgeBest = np.full(self.nEdges, np.inf)
        self.edgeCorrelation = np.ones(self.nEdges, dtype=float) 
        self.edgeProb = np.empty(self.nEdges, dtype=float)
        self.edgeNum = np.ones(self.nEdges, dtype=int) #number of vertices this particular transition goes through, one at the beginning
        self.rng = np.random.default_rng()
        self.decay = decay
        return
    
    def offsetProbMatrix(self):
        minVal = np.min(self.edgeCorrelation)
        self.edgeProb = self.edgeCorrelation + 1.1*abs(minVal)
        total = np.sum(self.edgeProb)
        self.edgeProb = np.divide(self.edgeProb, total)
        self.edgeCorrelation *= self.decay
        return
    
    def getValidTransitions(self, notValid):
        valid = []
        for i in range(self.nEdges):
            allow = True
            for j in range(self.edgeNum[i]):
                if self.edges[i][j] in notValid:
                    allow = False
                    break
            if allow:
                valid.append(i)

        return valid

    def chooseTransition(self, notValid):
        valid = self.getValidTransitions(notValid)
        if len(valid) == 0:
            print("no valid found")
            exit()
        
        prob = self.edgeProb[valid]
        constant = 1/np.sum(prob)
        prob *= constant
        cur = self.rng.choice(valid,p=prob)
        return self.edges[cur], cur
    
    def getBestTransition(self, notValid):
        valid = self.getValidTransitions(notValid)
        if len(valid) == 0:
            print("no valid found")
            exit()
        
        best = valid[0]
        for i in valid:
            if self.edgeProb[i] > self.edgeProb[best]:
                best = i
    
        return self.edges[best]

    def updateEdgeVals(self, edgeID, score, normalisedVal) -> None:

        self.edgeCorrelation[edgeID] += normalisedVal
        if score < self.edgeBest[edgeID]:
            self.edgeBest[edgeID] = score
        return

    def addEdge(self):
        
        return
    
    def removeEdge(self):

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
            try:
                j = circuitVector[f]
                distance += self.distanceMatrix[i][j]
                i = j
            except:
                print(circuitVector)
                exit()

        return distance

class tester():
    def __init__(self, graph:Graph, iterCount:int, popSize:int, probDecay:float):
        self.verticeCount = len(graph.distanceMatrix)
        vertices = [[i] for i in range(self.verticeCount)]

        self.vertices = np.empty(self.verticeCount, dtype=object)
        for i in range(len(graph.distanceMatrix)):
            temp = vertices.copy()
            temp.pop(i)
            self.vertices[i] = vertex(temp, probDecay)

        self.graph = graph
        self.mean = 0
        self.solutionCount = 0 
        self.decay = probDecay
        self.iterCount = iterCount
        self.popSize = popSize

    def mainRun(self):
        for i in range(self.iterCount):
            self.offsetProbMatrix()
            solutions = np.empty((self.popSize, 2),dtype=object)
            for j in range(self.popSize):
                temp, tempID = self.genProbChromosome()
                solutions[j] = self.graph.calcDistance(temp), tempID
            self.batchCalc(solutions[:, 0])
            add = []
            rm = []
            for j in solutions:
                if j[0] < self.mean - (2*self.sd):
                    add.append((j[0], j[1]))
                elif j[0] > self.mean + (2*self.sd):
                    rm.append((j[0], j[1]))
            
                normalisedVal = self.getNormalisedVal(j[0])
                self.updateCorrelation(j[1], j[0],normalisedVal)
            
            for sol in add:
                print(sol)
                #solutionID format: [(vertexID, edgeID),...,...]
                self.addEdge(sol)
            self.genBestGuess()
            print(i)
        
        for vertex in self.vertices:
            print(vertex.edgeCorrelation)
        self.genBestGuess()
        return
    
    def addEdge(self, solution):
        weights =[]
        for pair in solution[1]:        
            weight = self.vertices[pair[0]].edgeBest[pair[1]]
            print(weight)
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
        for vertex in self.vertices:
            vertex.offsetProbMatrix()

    def getNormalisedVal(self, score:float):
        p = 1
        a = 10
        temp = (math.e)**(-(((score-self.mean)**2)/(2*(self.sd)**2))**p)
        val = a - a*temp
        if score > self.mean:
            val = -abs(val)
        return val

    def updateCorrelation(self, solutionID, score, normalisedVal):
        #solutionID format: [(vertexID, edgeID),...,...]
        


        for edge in solutionID:
            val = self.vertices[edge[0]].updateEdgeVals(edge[1], score, normalisedVal)
    
    def genBestGuess(self) -> list:
        cur = np.random.randint(0, self.verticeCount)
        bestGuess = [cur]

        count = 1
        while count < self.verticeCount:
            edges = self.vertices[cur].getBestTransition(bestGuess)
            for edge in edges:
                bestGuess.append(edge)
                count += 1

            cur = edges[-1]

        print(bestGuess)
        print("best guess: {} has value {} \n".format(bestGuess, self.graph.calcDistance(bestGuess)))
        print("mean: {}  sd: {} \n".format(self.mean, self.sd))
        return

    def genProbChromosome(self):
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

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)

def main():
    
    graph = fetchGraph('burma14.xml')
    agent = tester(graph, 100, 50, 0.95)
    agent.mainRun()
    # convert array into dataframe 
    """  DF = pd.DataFrame(agent.minLenMatrix) 
    DF2 = pd.DataFrame(graph.distanceMatrix)
    DF3 = pd.DataFrame(agent.correlationMatrix)
    DF4 = pd.DataFrame(agent.countMatrix)
    # save the dataframe as a csv file 
    DF.to_csv("data1.csv")
    DF2.to_csv("data2.csv")
    DF3.to_csv("correlationMatrix.csv")
    DF4.to_csv("countMatrix.csv") 
    """
    return

if __name__ == "__main__":
    main()