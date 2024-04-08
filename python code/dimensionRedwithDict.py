
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



Maintain genetic diversity:
    seperate probability of choosing edge from correlation : Some runs never find best edge, other runs once its found find it dozens of times.
    falling into local minima of decent but not the best edge... 
    force either a probability based on previous edge usage - less used edges are more favourable, make normalisation function less harsh -> a from 10 to 1.

































"""

class vertex():
    def __init__(self, edges, decay, offSet) -> None:
        self.edges = {}
        for edge in edges:
            #key = vertexes edge travels to
            #value = [correlation, best]
            self.edges[edge] = [1, np.inf]
        self.nEdges = len(edges)
        self.rng = np.random.default_rng()
        self.decay = decay
        self.offSet = offSet
        return
    
    def offsetCorrelation(self):

        minVal = np.inf
        for key in self.edges.keys():
            if self.edges[key][0] < minVal:
                minVal = self.edges[key][0]

        if minVal < 0:
            for key in self.edges.keys():
                self.edges[key][0] = self.decay*(self.edges[key][0] + self.offSet*(abs(minVal)))
        else:
            for key in self.edges.keys():
                self.edges[key][0] = self.decay*(self.edges[key][0])
        return
    
    def getValidTransitions(self, notValid, remainingNodes):
        temp = notValid.copy()
        temp.pop(0)
        valid = []
        for key in self.edges.keys():
            allow = True

            if len(key) == remainingNodes:

                for node in key:
                    if node in temp:
                        allow = False
                        break
            else:
                for node in key:
                    if node in notValid:
                        allow = False
                        break
            if allow:
                valid.append(key)

        return valid

    def chooseTransition(self, notValid, remainingNodes):
        valid = self.getValidTransitions(notValid, remainingNodes)
        if len(valid) == 0:
            print("no valid found")
            exit()

        total = 0
        for key in valid:
            total += self.edges[key][0]
        
        val = np.random.uniform(0, total)

        total = 0
        for key in valid:
            total += self.edges[key][0]
            if total > val:
                return key
        
        print("something went wrong")
        exit()
    
    def getBestTransition(self, notValid, remainingNodes):
        valid = self.getValidTransitions(notValid, remainingNodes)
        if len(valid) == 0:
            print("no valid found")
            exit()
        
        best = valid[0]
        for key in valid:
            if self.edges[key][1] < self.edges[best][1]:
                best = key
    
        return best

    def updateEdgeVals(self, edgeID, score, normalisedVal) -> None:
        
        self.edges[edgeID][0] += normalisedVal
        if score < self.edges[edgeID][1]:
            self.edges[edgeID][1] = score
        return

    def addEdge(self, EdgeID, verticesToAdd):
        newEdge = []
        for i in EdgeID:
            newEdge.append(i)
        for i in verticesToAdd:
            newEdge.append(i)
        newKey = tuple(newEdge)
        self.edges[EdgeID][0] /= 2
        self.edges[newKey] = [self.edges[EdgeID][0],self.edges[EdgeID][1]]
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
            j = circuitVector[f]
            distance += self.distanceMatrix[i][j]
            i = j
           
        return distance

class tester():
    """
    value of probDecay, offset and popsize all affect each other with the outcome,
     i.e to be consistent when changing popsize, also change probDecay and offset
     as these are calculated not as a ratio of each population go, but on the sum of the total popsize go
    
    
    """

    def __init__(self, graph:Graph, iterCount:int, popSize:int, probDecay:float, offSet:float):
        self.verticeCount = len(graph.distanceMatrix)
        vertices = [(i,) for i in range(self.verticeCount)]
        self.vertices = np.empty(self.verticeCount, dtype=vertex)
        for i in range(len(graph.distanceMatrix)):
            temp = vertices.copy()
            temp.pop(i)
            self.vertices[i] = vertex(temp, probDecay, offSet)

        self.graph = graph
        self.mean = 0
        self.solutionCount = 0 
        self.decay = probDecay
        self.iterCount = iterCount
        self.popSize = popSize
        self.best = np.inf
        self.bestSol = None
        self.ratio = 0

    def mainRun(self):
        for i in range(self.iterCount):
            self.offsetProbMatrix()
            solutions = np.empty((self.popSize, 2),dtype=object)
            for j in range(self.popSize):
                temp, tempID = self.genProbChromosome()
                solutions[j] = self.graph.calcDistance(temp), tempID
            self.batchCalc(solutions)
            add = []
            rm = []
            print(self.best)
            print(self.bestSol)

            for j in solutions:
                
                if j[0] < self.best + self.ratio*self.sd:
                    add.append(j[1])
                elif j[0] > self.mean + (2*self.sd):
                    rm.append(j[1])

                normalisedVal = self.calcSolUtility(j[0])
                self.updateCorrelation(j[1], j[0],normalisedVal)
            
            self.ratio = 0.5
            
            for sol in add:
                #solutionID format: [(vertexID, edgeID),...,...]
                self.addEdge(sol)
                
            self.genBestGuess()
            print(i)
        
        self.genBestGuess()
        return
    
    def normaliseEdges(self, solution):
        #vertexID, edgeID
        length = len(solution)
        normalisedVals = np.empty(length, dtype=float)
        for i in range(len(solution)):
            #solution[i][0] is the vertexID, solution[i][1] is the edgeID, the final [1] represents the edge dictionary stored best value
            normalisedVals[i] = self.vertices[solution[i][0]].edges[solution[i][1]][1]

        minVal = np.min(normalisedVals)
        difference = np.max(normalisedVals) - minVal
        if difference == 0:
            normalisedVals = np.full(length, 0.5)
            return normalisedVals
        
        normalise = lambda x: (x-minVal)/difference
        normalisedVals = normalise(normalisedVals)

        return normalisedVals
    
    def removeEdge(self, solution):
        normalisedVals = self.normaliseEdges(solution)


    def addEdge(self, solution):
        #vertexID, edgeID
        normalisedVals = self.normaliseEdges(solution)
        if len(normalisedVals) != len(solution):
            print("error, add edge normal values not equal to sol")
            exit()
        j = 0
        for i in range(1, len(normalisedVals)):
            if normalisedVals[j] < 0.3 and normalisedVals[i] < 0.3:
                self.vertices[solution[j][0]].addEdge(solution[j][1], solution[i][1])
            j = i
        return

    def batchCalc(self, solutions):
        tempMean = 0
        self.variance = 0
        self.meanAbsoluteDeviation = 0
        length = len(solutions)

        for i in range(length):
            sol = solutions[i, 0]
            if sol < self.best:
                self.best = sol
                self.bestSol = solutions[i, 1]

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
            vertex.offsetCorrelation()

    def calcSolUtility(self, score:float):
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
            self.vertices[edge[0]].updateEdgeVals(edge[1], score, normalisedVal)
    
    def genBestGuess(self) -> list:
        cur = np.random.randint(0, self.verticeCount)
        bestGuess = [cur]

        count = 0
        remainingNodes = self.verticeCount
        temp = []
        while count < self.verticeCount:
            edges = self.vertices[cur].getBestTransition(bestGuess, remainingNodes)
            for edge in edges:
                bestGuess.append(edge)
                count += 1
                remainingNodes -= 1
            temp.append(edges)
            cur = edges[-1]
        print(temp)
        print("best guess: {} has value {} \n".format(bestGuess, self.graph.calcDistance(bestGuess)))
        print("mean: {}  sd: {} \n".format(self.mean, self.sd))
        return

    def genProbChromosome(self):
        cur = np.random.randint(0, self.verticeCount)
        solution = []
        solution.append(cur)
        solutionID = [] #vertexID, edgeID
        count = 0
        remainingNodes = self.verticeCount
        while count < self.verticeCount:
            edges = self.vertices[cur].chooseTransition(solution, remainingNodes)
            for edge in edges:
                solution.append(edge)
                count += 1
                remainingNodes -= 1
            solutionID.append((cur, edges))
            cur = edges[-1]
        solution.pop(0)

        return solution, solutionID

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)

def main():
    
    graph = fetchGraph('burma14.xml')
    agent = tester(graph, 100, 50, 0.975, 1.025)
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