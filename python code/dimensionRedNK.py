
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



                            experiments : try reward function of reinforcement learning: 1/value 
                                            try adding RL epsilon -> when calcing prob rather than alpha* corr + beta * best, 
                                            can be epsilon % chance of being 100% corr, then 1-epsilon for best

"""

class Node():
    def __init__(self, depth, momentum):
        self.branches = []
        self.probTotal = 0
        self.momentum = momentum
        self.depth = depth
        self.prob = 0
        self.transitionProb = 0
        
    def updateCorrelation(self,  score, corrVal):
        pass

    def addBranch(self, transition) -> None:
        newBranch = TreeNode(self, transition)
        self.Branches.append(newBranch)
        return
    
    def chooseTransition(self, solution, blankVal):
        valid = []
        path = []
        total = self.transitionProb
        for branch in self.branches:
            if solution[branch.transition.dimension] == blankVal:
                total += branch.transitionProb
                valid.append(branch)

        val = np.random.uniform(0, total)

        total = 0
        key = self
        for branch in valid:
            total += branch.prob
            if total > val:
                path, key = branch.chooseTransition()
                break
        return path ,key
    
    def calcProb(self, TransitionProb):
        self.transitionProb = self.momentum*TransitionProb
        """
            depth dropOff = (1/depth)^theta, where theta is between 0 and 1
            low theta doesnt punish depth, high theta punishes depth a lot
            ((1/depth)^theta) can be calculated on __init__()

        self.prob = self.momentum*((1/depth)^theta)*transitionProb
        
        """
        self.prob = self.transitionProb
        for branch in self.branches:
            self.prob += branch.calcProb()

        return self.prob

class TreeNode(Node):
    def __init__(self, parent, transition, momentum) -> None:
        super().__init__(depth = parent.depth + 1, momentum=momentum)
        self.parent = parent
        self.transition = transition
        pass

    def calcProb(self) -> None:
        return super().calcProb(self.transition.transitionProb)
    
    def addBranch(self, transition) -> None:
        super().addBranch(transition)

    def chooseTransition(self, solution, blankVal):
        path, key = super().chooseTransition(solution, blankVal)
        path.append((self.transition.dimension, self.transition.val))
        return path, key
    
    def updateCorrelation(self, score, corrVal, momentumMultiplier = 1):
        self.transition.updateCorrelation(score, corrVal, momentumMultiplier)
        self.parent.updateCorrelation(score, corrVal, momentumMultiplier)

        pass
    
class Transition(Node):
    def __init__(self, val, dimension) -> None:
        super().__init__(depth=1, momentum=1)
        self.best = np.inf
        self.correlation = 0
        self.val = val
        self.dimension = dimension

    def calcProb(self) -> None:
        return super().calcProb(self.transitionProb)
    
    def addBranch(self, transition) -> None:
        super().addBranch(transition)

    def chooseTransition(self, solution, blankVal):
        path, key = super().chooseTransition(solution, blankVal)
        path.append((self.dimension, self.val))
        return path, key

    def calcSelfProb(self, correlationMin, correlationDifference, bestMin, bestDifference, alpha, beta):
        corTemp = (self.correlation - correlationMin)/correlationDifference 
        bestTemp = (self.best - bestMin)/bestDifference
        self.transitionProb = alpha*corTemp + beta*bestTemp
        return
    
    def updateCorrelation(self, score, corrVal, momentumMultiplier = 1):
        """
        try multiple mometum multipliers
        
        """
        self.correlation += corrVal
        
        if score < self.best:
            self.best = score
            self.momentum 
            return True
        return False
    
    def updateMometum(self, momentumDif):
        self.momentum += momentumDif
        
        


class vertex():
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
            self.deletedReq.append(sum(edgeID))
        return
    
class TransitionMatrixTree():
    """
    value of probDecay, offset and popsize all affect each other with the outcome,
     i.e to be consistent when changing popsize, also change probDecay and offset
     as these are calculated not as a ratio of each population go, but on the sum of the total popsize go
    
    
    """

    def __init__(self, optimisationFunc, posTransitions, Ndimen,iterCount:int, popSize:int, probDecay:float, offSet:float, maxTreeDepth, alpha, beta):
        
        self.optimisationFunc = optimisationFunc
        if offSet <= 1:
            print("offset must be greater than 1")
            exit()
        #posTransitions is array of the transitions
        self.possibleTransitions = posTransitions  
        #number of dimensions      
        self.Dimensions = Ndimen
        self.transitionMatrix = [[Transition(j, i, maxTreeDepth, alpha, beta) for j in range(self.possibleTransitions)] for i in range(self.Dimensions)]
        self.probMatrix = np.empty[(self.Dimensions, len(self.possibleTransitions))]
        self.transitionMatrix = np.empty[(self.Dimensions, len(self.possibleTransitions))]
        for i in range(self.Dimensions):
            for j in range(len(self.possibleTransitions)):
                self.transitionMatrix[i,j] = Transition(self.possibleTransitions[j], i, maxTreeDepth, alpha, beta)

        self.mean = 0
        self.decay = probDecay
        self.iterCount = iterCount
        self.popSize = popSize
        self.best = np.inf
        self.bestSol = None
        self.ratio = 0
        self.prevVal = np.inf
        self.prevBest = []
        self.alpha = alpha
        self.beta = beta
        self.maxTreeDepth = maxTreeDepth

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

        for i in range(self.iterCount):
            beginLoop = time.perf_counter()
            self.CalcProbMatrix()
            afterOffset = time.perf_counter()
            offsetCor = afterOffset - beginLoop

            solutions = np.empty((self.popSize, 2),dtype=object)
            for j in range(self.popSize):
                sol, solID = self.genNKSolution()
                solutions[j] = self.optimisationFunc.calcScore(sol), solID
            afterGenSol = time.perf_counter()
            genSol =  afterGenSol - afterOffset

            self.batchCalc(solutions)
            afterBatchCalc = time.perf_counter()
            batchCalcT = afterBatchCalc - afterGenSol

            add = []
            rm = []
            print(self.best)
            
            for sol in solutions:
                
                if sol[0] < bestRatio*self.best + meanRatio*self.mean:
                    add.append(sol)
                elif sol[0] > self.mean + (removeRatio*self.sd):
                    rm.append(sol[1])
                
                reward = self.calcReward(sol[0])
                self.updateCorrelation(sol[1], sol[0], reward)
            
            afterCorrelationCalc = time.perf_counter()
            correlationCalc = afterCorrelationCalc - afterBatchCalc

            bestRatio += 0.6/self.iterCount
            meanRatio -= 0.6/self.iterCount
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
            normalisedVals = self.normaliseEdges(sol)
            if len(normalisedVals) != len(sol):
                print("error, add edge normal values not equal to sol")
                exit()

            for i in range(len(normalisedVals)):
                if normalisedVals[i] > 0.75:
                    toDelete.add((sol[i][0], sol[i][1]))
        
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
    
    def CalcProbMatrix(self):
        minCorr = np.inf
        maxCorr = 0
        minBest = np.inf
        maxBest = 0
        for i in range(len(self.transitionMatrix)):
            for j in range(len(self.transitionMatrix[0])):
                if self.transitionMatrix[i][j].correlation > maxCorr:
                    maxCorr = self.transitionMatrix[i][j].correlation
                elif self.transitionMatrix[i][j].correlation < minCorr:
                    minCorr = self.transitionMatrix[i][j].correlation

                if self.transitionMatrix[i][j].best > maxBest:
                    maxBest = self.transitionMatrix[i][j].correlation
                elif self.transitionMatrix[i][j].best < minBest:
                    minBest = self.transitionMatrix[i][j].correlation

        corrDifference = maxCorr - minCorr
        bestDifference = maxBest - minBest

        for i in range(len(self.transitionMatrix)):
            for j in range(len(self.transitionMatrix[0])):
                self.transitionMatrix[i][j].calcSelfProb(minCorr, corrDifference, minBest, bestDifference, self.alpha, self.beta)

        for i in range(len(self.transitionMatrix)):
            for j in range(len(self.transitionMatrix[0])):
                prob = self.transitionMatrix[i][j].calcProb()
                self.probMatrix[i, j] = prob
        return
    
    def calcReward(self, score:float):
        p = 1
        a = 1
        temp = (math.e)**(-(((score-self.mean)**2)/(2*(self.sd)**2))**p)
        val = a - a*temp
        if score > self.mean:
            val = -abs(val)
        return val

    def updateCorrelation(self, solutionID, score, reward):
        #solutionID format: [(vertexID, edgeID),...,...]
        for edge in solutionID:
            edge.updateCorrelation(score, reward)

    def genNKSolution(self):
        blankVal = 2
        solutionID = []
        solution = np.fill((self.Dimensions),blankVal)

        for dimension in range(self.Dimensions):
            if solution[dimension] != blankVal:
                continue
            cur = None
            total = np.sum(self.probMatrix[dimension,:])
            val = np.random.uniform(0, total)
            total = 0
            for transition in range(len(self.probMatrix[dimension, :])):
                total += self.probMatrix[dimension, transition]
                if total > val:
                    cur = transition
                    break
            
            edges, edgeID = self.transitionMatrix[dimension, transition].chooseTransition(solution, blankVal)
            solutionID.append((edgeID))
            for edge in edges:
                #edges formated as [(dimension, value),...]
                solution[edge[0]] = edge[1]

        if blankVal in solution:
            print("generated solution does not contain all nodes")
            exit()

        return solution, solutionID

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)

def main():
    
    graph = fetchGraph('gr202.xml')
    #iter count, popsize, correlation decay, offset: >1
    print("got graph")
    agent = TransitionMatrixTree(graph, 500, 250, 0.875, 1.125, 50)
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