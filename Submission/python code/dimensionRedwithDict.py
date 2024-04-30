
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






Justify choice of using correlation for probability, over combined heuristic of best edge probability and usage? test out options.


have a stack for when generating solution -> help with add edge and remove edge specifically
    creating graph of possible solutions impractical -> have stack which goes and goes until no possible solutions
                                                        pop latest addition and add to impossible pile and try again
                                                        
                                                        reduce correlation? -> prob of popped edges? 
                                                        depending on depth -> i.e if 10 nodes added, and 11th is impossible, pop 10th node go back to 9th 
                                                        and try 10th node again, if all 10th's nodes are impossible for that then pop 9th and go to 8th
                                                        9th edge which was impossible for all nodes more punished than individual 10th's nodes.

                                                        test if punish all nodes tested or just the 1st one to cause an issue.
                                                        i.e. if 2nd edge caused the problem and we only find out at the 9th edge, punish all edges tried between 2nd and 9th
                                                        or only 2nd majorly punished?

                                                        implicitly removing an edge, and then a combo edge added which forces you to use that edge isnt a great combo edge
                                                        as it forces you to use a bad (removed) edge so should be punished 

                            option 2:
                                        just add the temporary edge: -> if fundamental edges (length 1) are removed, turn their correlation to 0, etc, and make it unable to
                                        change -> or just add it temporarily if needed. 

                                        possibly then also give a punishment to all of the edges necessary in the solution so far, allows for computationally cheap solution
                                        enables chance of exploration and gives a negative for using a bad edge.


                            option 2 more viable for problem where sampling isnt an issue? -> for more expensive sampling option one wins out
























"""
 

class vertex():
    def __init__(self, edges, decay, offSet, ID, deletedReq, parent) -> None:
        self.edges = {}
        self.deletedReq = []
        for edge in deletedReq:
            self.deletedReq.append((edge,))

        for edge in edges:
            #key = vertexes edge travels to
            #value = [correlation, best, momentum]
            self.edges[(edge,)] = [1, np.inf, 0]
        self.nEdges = len(edges)
        self.rng = np.random.default_rng()
        self.decay = decay
        self.offSet = offSet
        self.ID = ID
        self.parent = parent
        return
    
    def offsetCorrelation(self):
        minVal = np.inf
        for key in self.edges.keys():
            if self.edges[key][0] < minVal:
                minVal = self.edges[key][0]

        if minVal < 0:
            for key in self.edges.keys():
                self.edges[key][0] = self.decay*(self.edges[key][0] + self.offSet*(abs(minVal)))
        elif minVal == 0:
            minVal = 0.2
            for key in self.edges.keys():
                self.edges[key][0] = self.decay*(self.edges[key][0] + self.offSet*(abs(minVal)))
        else:
            for key in self.edges.keys():
                self.edges[key][0] = self.decay*(self.edges[key][0])
        return
    
    def getValidTransitions(self, notValid, remainingNodes):
        last = notValid[0]
        valid = []

        for key in self.edges.keys():
            allow = True

            if len(key) == remainingNodes:
                if key[-1] == last:
                    for node in key:
                        if node in notValid and node != last:
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

    def isTransitionValid(self, keyTuple, notValid, remainingNodes):
        last = notValid[0]
        key = keyTuple[0]

        if remainingNodes == 1:
            if key == last:
                return True
            return False
        
        if key in notValid:
            return False

        return True

    def chooseTransition(self, notValid, remainingNodes):
        valid = self.getValidTransitions(notValid, remainingNodes)

        if len(valid) == 0:
            key = np.random.randint(0, len(self.deletedReq))
            for i in range(len(self.deletedReq)):
                key = (key+1)%len(self.deletedReq)
                isValid = self.isTransitionValid(self.deletedReq[key], notValid, remainingNodes)
               
                if isValid:
                    self.edges[self.deletedReq[key]] = [0 ,self.parent.mean]
                    return self.deletedReq[key], True
                
            print("no deleted node found error")
            exit()

        
        total = 0
        for key in valid:
            total += self.edges[key][0]
        
        if total == 0:
            key = np.random.randint(0, len(valid))
            return valid[key], False

        val = np.random.uniform(0, total)

        total = 0
        for key in valid:
            total += self.edges[key][0]
            if total > val:
                return key, False
        
        print("something went wrong")
        exit()

    def updateEdgeVals(self, edgeID, score, normalisedVal) -> None:
        
        self.edges[edgeID][0] += normalisedVal
        if score < self.edges[edgeID][1]:
            self.edges[edgeID][1] = score
        return
    
    def penaliseEdgeVals(self, edgeID) -> None:
        self.edges[edgeID][0] *= 0.8
    
    def addEdge(self, EdgeID, verticesToAdd, best):
        newEdge = []
        for i in EdgeID:
            newEdge.append(i)
   
        for i in verticesToAdd:
            newEdge.append(i)

        newKey = tuple(newEdge)

        if newKey in self.edges:
            if best < self.edges[newKey][1]:
                self.edges[newKey][1] = best
            return
        self.edges[newKey] = [(self.edges[EdgeID][0]/2),best]
        self.edges[EdgeID][0] /= 1.5
        return 
    
    def removeEdge(self, edgeID):
        self.edges.pop(edgeID)
        #edges of length one are needed in order to generate solutions in certain cases, add to special list to only consider
        #in worst cases when no other valid edges exist
        if len(edgeID) == 1:
            self.deletedReq.append(edgeID)
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

    def __init__(self, graph:Graph, iterCount:int, popSize:int, probDecay:float, offSet:float, numberEdges):
        self.graph = graph
        if offSet <= 1:
            print("offset must be greater than 1")
            exit()

        self.verticeCount = len(graph.distanceMatrix)
        self.vertices = np.empty(self.verticeCount, dtype=vertex)
        index = np.arange(self.verticeCount)
        for i in range(self.verticeCount):
            ordered = np.column_stack((index, self.graph.distanceMatrix[i]))
            ordered = np.rint(ordered[ordered[:, 1].argsort(), 0]).astype(int)
            self.vertices[i] = vertex(ordered[0:numberEdges], probDecay, offSet, i, ordered[numberEdges:], self)

        self.mean = 0
        self.solutionCount = 0 
        self.decay = probDecay
        self.iterCount = iterCount
        self.popSize = popSize
        self.best = np.inf
        self.bestSol = None
        self.ratio = 0
        self.prevVal = np.inf
        self.prevBest = []
        self.maxEdgeLen = 6

        self.offsetCorrelationTime = np.empty(iterCount,dtype=float)
        self.genSolsTime = np.empty(iterCount,dtype=float)
        self.batchCalcTime = np.empty(iterCount,dtype=float)
        self.correlationUpdateTime = np.empty(iterCount,dtype=float)
        self.afterRemoveAndDeleteTime = np.empty(iterCount,dtype=float)
        self.bestSolMatrix = np.empty(iterCount,dtype=float)
        self.nAdded = np.empty(iterCount,dtype=float)
        self.nRemoved = np.empty(iterCount,dtype=float)
        self.meanOverTime = np.empty(iterCount,dtype=float)
        self.sdOverTime = np.empty(iterCount,dtype=float) 

    def mainRun(self):
        bestRatio = 0.3
        meanRatio = 0.7
        removeRatio = 3
        for i in range(self.iterCount):
            beginLoop = time.perf_counter()
            self.offsetProbMatrix()
            afterOffset = time.perf_counter()
            offsetCor = afterOffset - beginLoop

            solutions = np.empty((self.popSize, 2),dtype=object)
            for j in range(self.popSize):
                temp, tempID = self.genProbChromosome()
                solutions[j] = self.graph.calcDistance(temp), tempID
            afterGenSol = time.perf_counter()
            genSol =  afterGenSol - afterOffset

            self.batchCalc(solutions)
            afterBatchCalc = time.perf_counter()
            batchCalcT = afterBatchCalc - afterGenSol

            add = []
            rm = []
            print(self.best)
            
            for j in solutions:
                
                """ if j[0] < bestRatio*self.best + meanRatio*self.mean:
                    add.append(j)
                elif j[0] > self.mean + (removeRatio*self.sd):
                    rm.append(j[1]) """
                
                if j[0] < 1*self.best + 1.2*self.sd:
                    add.append(j)
                elif j[0] > self.mean + (2*self.sd):
                    rm.append(j[1])
                

                normalisedVal = self.calcSolUtility(j[0])
                self.updateCorrelation(j[1], j[0],normalisedVal)
            
            afterCorrelationCalc = time.perf_counter()
            correlationCalc = afterCorrelationCalc - afterBatchCalc

            bestRatio += 0.65/self.iterCount
            meanRatio -= 0.65/self.iterCount
            removeRatio -= 2/self.iterCount
            self.addEdges(add)
            self.removeEdges(rm)
            afterRemoveAndAdd =  time.perf_counter()
            rmAndAdd = afterRemoveAndAdd - afterCorrelationCalc
            print("removed: ", len(rm))
            print("added: ", len(add))
            print("mean: ", self.mean)
            print("sd: ", self.sd)
            #self.genBestGuess()
            print("offset Correlation: ", offsetCor)
            print("generate solutions: ", genSol)
            print("Batch calc: ", batchCalcT)
            print("Correlation Update: ", correlationCalc)
            print("Add and delete edges: ", rmAndAdd)
            self.offsetCorrelationTime[i] = offsetCor
            self.genSolsTime[i] = genSol
            self.batchCalcTime[i] = batchCalcT
            self.correlationUpdateTime[i] = correlationCalc
            self.afterRemoveAndDeleteTime[i] = rmAndAdd
            self.bestSolMatrix[i] = self.best
            self.nAdded[i] = len(add)
            self.nRemoved[i] = len(rm)
            self.meanOverTime[i] = self.mean
            self.sdOverTime[i] = self.sd
            print(i)
        
        print("--------------------------------\n")
        print(self.best)
        #print(self.bestSol)
        print("\n--------------------------------")
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
    
    def removeEdges(self, solutions):
        toDelete = set()
        for sol in solutions:
            #vertexID, edgeID          
            try:             
                normalisedVals = self.normaliseEdges(sol)
                if len(normalisedVals) != len(sol):
                    print("error, add edge normal values not equal to sol")
                    exit()

                for i in range(len(normalisedVals)):
                    if normalisedVals[i] > 0.75:
                        toDelete.add((sol[i][0], sol[i][1]))
            except:
                print("sol:", sol)
            
        for edge in toDelete:
            self.vertices[edge[0]].removeEdge(edge[1])
        return
  
    def addEdges(self, solutions):
        for sol in solutions:
            solution = sol[1]
            score = sol[0]
        
            #vertexID, edgeID      
            normalisedVals = self.normaliseEdges(solution)
            if len(normalisedVals) != len(solution):
                print("error, add edge normal values not equal to sol")
                exit()

            elif (normalisedVals==0.5).all() == True :
                j = 0
                for i in range(1, len(normalisedVals)):
                    self.vertices[solution[j][0]].addEdge(solution[j][1], solution[i][1], score)

                    j = i
                return

            j = 0
            for i in range(1, len(normalisedVals)):
                if normalisedVals[j] < 0.25 and normalisedVals[i] < 0.25:
                    if len(solution[j][1]) > self.maxEdgeLen or len(solution[i][1]) > self.maxEdgeLen:
                        continue
                    self.vertices[solution[j][0]].addEdge(solution[j][1], solution[i][1], score)

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

    def penaliseCorrelation(self, solutionID):
        for edge in solutionID:
            self.vertices[edge[0]].penaliseEdgeVals(edge[1])

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
            temp.append((cur, edges))
            cur = edges[-1]
        bestGuess.pop(0)
        
        print("solution format: ", temp)
        print("best guess: {} has value {} \n".format(bestGuess, self.graph.calcDistance(bestGuess)))
        print("mean: {}  sd: {} \n".format(self.mean, self.sd))
        return bestGuess

    def genProbChromosome(self):
        cur = np.random.randint(0, self.verticeCount)
        solution = []
        solution.append(cur)
        solutionID = [] #vertexID, edgeID
        count = 0
        remainingNodes = self.verticeCount
        while count < self.verticeCount:
            edges, isDeleted = self.vertices[cur].chooseTransition(solution, remainingNodes)
            for edge in edges:
                solution.append(edge)
                count += 1
                remainingNodes -= 1

            solutionID.append((cur, edges))
            cur = edges[-1]
            if isDeleted:
                self.penaliseCorrelation(solutionID)

        temp = solution.pop(0)

        if len(solution) != self.verticeCount:
            print("generated solution does not contain all nodes")
            print(solution)
            print(solutionID)
            print(temp)
            print(count, remainingNodes)
            exit()

        return solution, solutionID

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)

def main():
    
    graph = fetchGraph('brazil58.xml')
    #iter count, popsize, correlation decay, offset: >1
    print("got graph")
    agent = tester(graph, 500, 150, 0.85, 1.2, len(graph.distanceMatrix[0]))
    print("got agent")

    agent.mainRun()
    # convert array into dataframe 

    
    ordered = np.column_stack((agent.offsetCorrelationTime, agent.genSolsTime, agent.batchCalcTime, agent.correlationUpdateTime, agent.afterRemoveAndDeleteTime, agent.bestSolMatrix, agent.nRemoved, agent.nAdded, agent.meanOverTime, agent.sdOverTime))
    columnNames = ["offsetCorrelationTime" , "genSolsTime", "batchCalcTime", "correlationUpdateTime", "afterRemoveAndDeleteTime", "bestSolMatrix", "nRemoved", "nAdded", "meanOverTime", "sdOverTime"]
    DF = pd.DataFrame(data =ordered, columns=columnNames) 
    # save the dataframe as a csv file 
    DF.to_csv("data1.csv")
    return

if __name__ == "__main__":
    main()