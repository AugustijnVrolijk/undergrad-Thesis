
import xml.etree.ElementTree as ET
import numpy as np
import math
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd 
import time


class Node():
    def __init__(self, depth, transitionMatrix, isRoot):
        self.branches = []
        self.depth = depth
        self.prob = 0
        self.transitionProb = 0
        self.beenExplored = 0
        self.transitionMatrix = transitionMatrix
        self.isRoot = isRoot


    def addBranch(self, branch, maxTreeDepth) -> None:
        if self.depth == maxTreeDepth:
            return
        
        if self.branchExists(branch):
            return
        newBranch = TreeNode(self, branch, False)
        self.branches.append(newBranch)
        return
    
    def branchExists(self, parent):
        for branch in self.branches:
            if branch.transition == parent:
                return True
        return False


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
    
    def chooseTSPTransition(self, solution, validSols, path):
        valid = []
        total = self.transitionProb
        
        if len(solution) < len(validSols):
            for branch in self.branches:
                isValid = True
                for val in solution:
                    if branch.transition.val == val:
                        isValid = False
                        break
                if isValid:
                    valid.append(branch)
                    total += branch.transitionProb
        else:
            for branch in self.branches:
                isValid = False
                for val in validSols:
                    if branch.transition.val == val:
                        isValid = True
                        break
                if isValid:
                    valid.append(branch)
                    total += branch.transitionProb

        val = np.random.uniform(0, total)

        total = 0
        key = self
        for branch in valid:
            total += branch.prob
            if total > val:
                path, key = branch.chooseTSPTransition(solution, validSols, path)
                break
        return path, key
    
    def calcProb(self, TransitionProb):
        self.transitionProb = TransitionProb
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
    def __init__(self, parent, transition, isRoot) -> None:
        super().__init__(depth = parent.depth + 1, transitionMatrix=parent.transitionMatrix,isRoot=isRoot)
        self.parent = parent
        self.transition = transition


    def calcProb(self) -> None:
        return super().calcProb(self.transition.transitionProb)
    
    def addBranch(self, branch, maxTreeDepth):
        super().addBranch(branch, maxTreeDepth)
        return
    
    def chooseTransition(self, solution, blankVal):
        path, key = super().chooseTransition(solution, blankVal)
        path.append((self.transition.dimension, self.transition.val))

        return path, key
    

    def chooseTSPTransition(self, solution, validSols, path):
        path.append((self.transition.dimension, self.transition.val))
        path, key = super().chooseTSPTransition(solution, validSols, path)
        return path, key
        
    def updateSelf(self, score, corrVal):
        
        isNewBest = self.transition.updateTransition(score, corrVal)
        self.parent.updateTransition(score, corrVal)
        return
    
    def updateTransition(self, score, corrVal):
        isNewBest = self.transition.updateTransition(score, corrVal)
        return
    """
     def updateSelf(self, score, corrVal, momentumMultiplier = 1):
        
        self.transition.updateSelf(score, corrVal, momentumMultiplier)
        momentumMultiplier = momentumMultiplier/2
        
        try multiple mometum multipliers values, currently at 1, halving by 2, but could be dividing by 1.5 etc..
        maybe linked to number of dimensions?
        
        isNewBest = self.parent.updateSelf(score, corrVal, momentumMultiplier)
        if isNewBest:
            self.momentum += 1
        else:
            val = self.momentum - momentumMultiplier*0.02
            if val > 0.02:
                self.momentum = val
        return isNewBest
    
        momentum shared between parents etc..
    
    """
    def getBest(self):
        return self.transition.best
    
    def getCorr(self):
        return self.transition.correlation
    
    def getVal(self):
        return self.transition.val

    
    def delete(self):
        for branch in self.branches:
            branch.delete()
            self.branches.remove(branch)
        del self
        return
    
    def getRoot(self):
        return self.parent.getRoot()
    
    def printSelf(self, noisy=True):
        print("branch with node:")
        self.transition.printSelf(noisy)
        print("with parent:")
        self.parent.printSelf(noisy)

    def delete(self):
        for branch in self.branches:
            branch.delete()
            self.branches.remove(branch)
        del self
        return
    
class Transition(Node):
    def __init__(self, val, dimension, transitionMatrix) -> None:
        super().__init__(depth=1, transitionMatrix=transitionMatrix,isRoot = True)
        self.best = np.inf
        self.correlation = 0
        self.val = int(val)
        self.dimension = int(dimension)
        
    def calcProb(self) -> None:
        return super().calcProb(self.transitionProb)
    
    def chooseTransition(self, solution, blankVal):
        path, key = super().chooseTransition(solution, blankVal)
        path.append((self.dimension, self.val))
        return path, key

    def chooseTSPTransition(self, solution, validSols, random):
        path = [(self.dimension, self.val)]

        if random:
            return path, self
        path, key = super().chooseTSPTransition(solution, validSols, path)
        return path, key

    def calcNumerator(self):
        corTemp = (self.correlation)**self.transitionMatrix.alpha
        bestTemp = ((1/self.best)**self.transitionMatrix.beta)
        if math.isnan(bestTemp):
            bestTemp = 0
        if math.isinf(bestTemp):
            bestTemp = 0
        
        self.Numerator = corTemp*bestTemp
        
        return self.Numerator
    
    def calcSelfProb(self, denum):      
        self.transitionProb = self.Numerator/denum
        return
    
    def updateSelf(self, score, corrVal):
        isNewBest = self.updateTransition(score, corrVal)
        return
    
    def updateTransition(self, score, corrVal):
        self.correlation += corrVal
        isNewBest = False
        if score < self.best:
            self.best = score
            isNewBest = True
        return isNewBest
    
    def getBest(self):
        return self.best
    
    def getCorr(self):
        return self.correlation
    
    def getVal(self):
        return self.val

    def removeChild(self, child):
        self.branches.remove(child)

    def getRoot(self):
        return self

    def addBranch(self, branch, maxTreeDepth):
        super().addBranch(branch, maxTreeDepth)
        return
    
    def delete(self):
        print("we are deleting root with val:", self.best)
        for branch in self.branches:
            branch.delete()
        #momentum set to 0 means it will never be called again
        return
    
    def printSelf(self, noisy=True):
        if noisy:
            return print("node: (dim:{},val:{}), has best: {}, corr: {}".format(self.dimension, self.val, self.best, self.correlation))
        else:
            return print("(dim:{},val:{})".format(self.dimension, self.val))
        
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

    def calcScore(self, circuitVector :list) -> int: 
        distance = 0
        i = circuitVector[-1]
        for f in range(len(circuitVector)):
            j = circuitVector[f]
            distance += self.distanceMatrix[i][j]
            i = j
           
        return distance


class TransitionMatrixTree():
    """
    value of probDecay, offset and popsize all affect each other with the outcome,
     i.e to be consistent when changing popsize, also change probDecay and offset
     as these are calculated not as a ratio of each population go, but on the sum of the total popsize go
    
    
    """

    def __init__(self, optimisationFunc, posTransitions, Ndimen,iterCount:int, popSize:int, maxTreeDepth, alpha, beta, corrRed, offSet, nAddBranch):
        
        self.optimisationFunc = optimisationFunc
        """ 
probDecay:float, offSet:float,
        if offSet <= 1:
            print("offset must be greater than 1")
            exit() """
        #posTransitions is array of the transitions
        self.possibleTransitions = posTransitions  
        #number of dimensions      
        self.Ndimen = Ndimen
        self.transitionMatrix = [[Transition(j, i, self) for j in self.possibleTransitions] for i in range(self.Ndimen)]
     
        self.probMatrix = np.empty((self.Ndimen, len(self.possibleTransitions)))


        #print('\n'.join([''.join(['({:2},{:2})'.format(item.val, item.dimension) for item in row]) for row in self.transitionMatrix]))
        self.mean = 0
        #self.decay = probDecay
        self.iterCount = iterCount
        self.popSize = popSize
        self.best = np.inf
        self.ratio = 0
        self.prevVal = np.inf
        self.prevBest = []
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1
        self.maxTreeDepth = maxTreeDepth
        self.corrRed = corrRed
        self.offSet = offSet
        self.nAddBranch = nAddBranch

        self.offsetCorrelationTime = np.empty(iterCount,dtype=float)
        self.genSolsTime = np.empty(iterCount,dtype=float)
        self.batchCalcTime = np.empty(iterCount,dtype=float)
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
                random = np.random.random()
                if random > self.epsilon:
                    sol, solID = self.genTSPSolution()
                else:
                    sol, solID = self.genTSPSolution(random=True)
                self.epsilon -= 0.001
                solutions[j] = self.optimisationFunc.calcScore(sol), solID
            afterGenSol = time.perf_counter()
            genSol =  afterGenSol - afterOffset

            self.batchCalc(solutions) #batchCalc contains normalisation calcs
            afterBatchCalc = time.perf_counter()
            batchCalcT = afterBatchCalc - afterGenSol
            
            add, rm = self.rankSolutions(solutions)
            #add = self.tournamentRankSols(solutions)
            #add, rm = self.heuristicRankSols(solutions, 0.7 ,0.3, removeRatio)
            self.addEdgesTSP(add)
            afterRemoveAndAdd =  time.perf_counter()
            rmAndAdd = afterRemoveAndAdd - afterBatchCalc
            self.offSetCorr()
            
            
            print("iteration: ", i)
            print("best: ", self.best)
            print("added: ", len(add))
            print("mean: ", self.mean)
            print("sd: ", self.sd)
            print("Time:\n offset Correlation: ", offsetCor)
            print("generate solutions: ", genSol)
            print("Batch calc: ", batchCalcT)
            print("Add and delete edges: ", rmAndAdd)
            self.offsetCorrelationTime[i] = offsetCor
            self.genSolsTime[i] = genSol
            self.batchCalcTime[i] = batchCalcT
            self.afterRemoveAndDeleteTime[i] = rmAndAdd
            self.bestSolMatrix[i] = self.best
            self.nAdded[i] = len(add)
            self.meanOverTime[i] = self.mean
            self.sdOverTime[i] = self.sd

            #print('\n'.join([''.join(['({:.5},{})'.format(item.prob, item.best) for item in row]) for row in self.transitionMatrix]))

        return self.best
    def printProb(self):
        print('\n'.join([''.join(['({:1.2f})'.format(self.probMatrix[i, j]) for i in range(self.Ndimen)]) for j in range(self.Ndimen)]))
        print("\n")

    def offSetCorr(self):
        
        for i in range(self.Ndimen):
            total = 0
            minVal = np.inf
            for edge in self.transitionMatrix[i]:
                temp = edge.correlation
                total += temp
                if temp < minVal:
                    minVal = temp
            minVal = abs(minVal)
            for edge in self.transitionMatrix[i]:
                edge.correlation = self.corrRed*edge.correlation + self.offSet*minVal
        return
    
    def tournamentRankSols(self, solutions):
        add = set()
        k = 10
        countTotal = 5
        if k < 2:
            print("tournament size too small")
            exit()

        while len(add) < countTotal:
            temp = np.empty((k, 2), dtype=object)

            for j in range(k):
                val = np.random.randint(0, 100)
                temp[j] = solutions[val]
            indices = np.argsort(temp[:, 0])
            val = indices[0] 
            add.add(tuple(temp[val, 1]))

        return add
    
    def rankSolutions(self, solutions):
        temp = np.argsort(solutions[:, 0])
        add = solutions[temp[0:self.nAddBranch], 1]
        rm = solutions[temp[(self.popSize-self.nAddBranch):], :]
        return add, rm

    def heuristicRankSols(self, solutions, bestRatio, addHeuristic, rmHeuristic):
        add = []
        rm = []
        for sol in solutions:
            #alternatively self.mean - addheurstic*self.sd
            if sol[0] < bestRatio*self.best + addHeuristic*self.mean:
                add.append(sol[1])
            elif sol[0] > self.mean + rmHeuristic*self.sd:
                rm.append(sol) 
        return add, rm
    
    def normaliseEdges(self, vals, inverse=False):
        #vertexID, edgeID

        normalisedVals = np.array(vals)
     
        minVal = np.min(normalisedVals)
        difference = np.max(normalisedVals) - minVal
        if difference == 0:
            normalisedVals = np.zeros(len(vals))
            return normalisedVals

        if inverse:
            normalise = lambda x: 1-((x-minVal)/difference)
        else:
            normalise = lambda x: (x-minVal)/difference
        normalisedVals = normalise(normalisedVals)

        return normalisedVals
    
    def removeEdges(self, solutions):
        toDelete = set()

        for temp in solutions:
            sol = temp[1]
            #EdegID
            temp = np.empty(len(sol))
            for i in range(len(sol)):
                temp[i] = sol[i].getCorr()
            """
            
            
            
            Why tf does this work with normaliseEdges(inverse= False) rather than inverse = True huh.
            anyway wtf...
            using correlation for normalisation - higher correlation is better, therefore want to inverse so higher corr values are not removed but idk
            """
            normalisedVals = self.normaliseEdges(temp, True)

            if len(normalisedVals) != len(sol):
                print("error, add edge normal values not equal to sol")
                exit()

            for i in range(len(normalisedVals)):
                if normalisedVals[i] > 0.7:
                    toDelete.add(sol[i])
        
        for edge in toDelete:
            edge.delete()
        return

    def addEdgesTSP(self, solutions):
        cutoff = 0.25
        branchesToAdd = {}
        for solution in solutions:
            temp = np.empty(len(solution))
            for i in range(len(solution)):
                temp[i] = solution[i].getBest()
            normalisedVals = self.normaliseEdges(temp)
            if len(normalisedVals) != len(solution):
                print("error, add edge normal values not equal to sol")
                exit()

            """
            adding edge threshold? gone for 0.25 but this can change
            
            """

            j = 0
            for i in range(1, len(solution)):
                if normalisedVals[j] <= cutoff and normalisedVals[i] <= cutoff:
                    if solution[j] not in branchesToAdd.keys():
                        branchesToAdd[solution[j]] = set()
                    branchesToAdd[solution[j]].add(solution[i].getRoot())
                j = i

        for key in branchesToAdd.keys():
            for branch in branchesToAdd[key]:
                key.addBranch(branch, self.maxTreeDepth)
            """
            add chance of mutation??? add branch to temp[val] but not the one given by temp[i]?"""

        return
  
    def addEdges(self, solutions):

        branchesToAdd = {}

        for solution in solutions:
            temp = np.empty(len(solution))
            for i in range(len(solution)):
                temp[i] = solution[i].getBest()
            normalisedVals = self.normaliseEdges(temp)     
            if len(normalisedVals) != len(solution):
                print("error, add edge normal values not equal to sol")
                exit()

            """
            adding edge threshold? gone for 0.25 but this can change
            
            """
            temp = []
            for i in range(len(solution)):
                if normalisedVals[i] < 0.25:
                    temp.append(solution[i])

            if len(temp) <= 1:
                continue

            """
            check condition if odd number - this will give preferable treatment to earlier dimensions -> 
            the first in the list temp will always get paired up with the second as it descends etc...
            jump by 2?"""
            for i in range(0 ,2*len(temp)):
                val1 = np.random.randint(len(temp))
                val2 = np.random.randint(len(temp))
                while val2 == val1:
                    val2 = np.random.randint(len(temp))

                if val1 < val2:
                    if temp[val1] not in branchesToAdd.keys():
                        branchesToAdd[temp[val1]] = set()
                    branchesToAdd[temp[val1]].add(temp[val2].getRoot())
                else:
                    if temp[val2] not in branchesToAdd.keys():
                        branchesToAdd[temp[val2]] = set()
                    branchesToAdd[temp[val2]].add(temp[val1].getRoot())

        for key in branchesToAdd.keys():
            for branch in branchesToAdd[key]:
                key.addBranch(branch, self.maxTreeDepth)
            """
            add chance of mutation??? add branch to temp[val] but not the one given by temp[i]?"""

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

        for i in range(length): 
            #self.updateQLearningCorrelation(solutions[i, 1], solutions[i, 0])
            self.updateGaussianCorrelation(solutions[i, 1], solutions[i, 0])
        return
    
    def CalcProbMatrix(self):
        for i in range(len(self.transitionMatrix)):
            total = 0
            for j in range(len(self.transitionMatrix[0])):
                total += self.transitionMatrix[i][j].calcNumerator()
            if total == 0:
                total = 1
            for j in range(len(self.transitionMatrix[0])):
                self.transitionMatrix[i][j].calcSelfProb(total)

        for i in range(len(self.transitionMatrix)):
            for j in range(len(self.transitionMatrix[0])):
                prob = self.transitionMatrix[i][j].calcProb()
                self.probMatrix[i, j] = prob
        return
    
    def calcGaussianReward(self, score:float):
        p = 0.7
        a = 1
        temp = (math.e)**(-(((score-self.mean)**2)/(2*(self.sd)**2))**p)
        val = a - a*temp
        if score > self.mean:
            val = -abs(val)
        val = 1/score
        return val
    
    def calcQlearningReward(self, score:float, edge:Node,learningRate, discountRate):
        lr = 0.7 #learning rate
        dr = 0.5 #discount rate
        reward =  lr*(-score + dr*(self.getBestNext(edge.getVal()))- edge.getCorr())     
        return reward

    def getBestNext(self, dimension):
        best = self.transitionMatrix[dimension][0].correlation
        for edge in self.transitionMatrix[dimension]:
            if edge.correlation > best:
                best = edge.correlation
        return best

    def updateGaussianCorrelation(self, solutionID, score):
        #solutionID format: [(vertexID, edgeID),...,...]
        reward = self.calcGaussianReward(score)
        for edge in solutionID:    
            edge.updateSelf(score, reward)
    
    def updateQLearningCorrelation(self, solutionID, score):
        #solutionID format: [(vertexID, edgeID),...,...]
        for edge in solutionID:
            reward = self.calcQlearningReward(score, edge, 0.3, 0.6)
            edge.updateSelf(score, reward)

    def genTSPSolution(self, random= False):

        solutionID = []
        validSols = np.arange(self.Ndimen).tolist()
        curDim = np.random.randint(0,self.Ndimen)
        validSols.remove(curDim)
        solution = [curDim]
        visited = 1

        while visited < self.Ndimen:
            
            total = np.sum(self.probMatrix[curDim,validSols])
                
            if total != 0 and not random:
                val = np.random.uniform(0, total)
                total = 0
                for transition in validSols:
                    total += self.probMatrix[curDim, transition]
                    #>= equal needed incase probability is 0 for all transitions
                    if total >= val:
                        cur = transition
                        break
            else:
                cur = np.random.choice(validSols)

            edges, edgeID = self.transitionMatrix[curDim][cur].chooseTSPTransition(solution, validSols, random)
            solutionID.append((edgeID))
            for i in range(len(edges)):
                curDim = edges[i][1]
                solution.append(curDim)
                #edges formated as [(dimension, value),...]
                visited += 1
                try:
                    validSols.remove(curDim)
                except:
                    print("failed to remove curDim")
                    print("edges:", edges)
                    print("sol:" ,solution)
                    print("cur: ", curDim)

                    print("valid:", validSols)
                    print("solID:")
                    for sol in solutionID:
                        sol.printSelf(False)
                    exit()

        solutionID.append(self.transitionMatrix[curDim][solution[0]])

        if len(solution) != self.Ndimen:
            print("---------------------------\ngen sol too short or long\n---------------------------")
            print("sol:" ,solution)
            print("cur: ", curDim)

            print("valid:", validSols)
            print("solID:")
            for sol in solutionID:
                sol.printSelf(False)
            exit()

        return solution, solutionID
    
    def genNKSolution(self):
        blankVal = 2
        solutionID = []
        solution = np.fill((self.Ndimen),blankVal)

        for dimension in range(self.Ndimen):
            if solution[dimension] != blankVal:
                continue
            cur = None
            total = np.sum(self.probMatrix[dimension,:])
            val = np.random.uniform(0, total)
            total = 0
            for transition in range(len(self.probMatrix[dimension, :])):
                total += self.probMatrix[dimension, transition]
                #>= equal needed incase probability is 0 for all transitions
                if total >= val:
                    cur = transition
                    break
            
            edges, edgeID = self.transitionMatrix[dimension][transition].chooseTransition(solution, blankVal)
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
    
    graph = fetchGraph('burma14.xml')
    #iter count, popsize, correlation decay, offset: >1
    print("got graph")
    temp = []
    for i in range(10):
        posTransitions = np.arange(len(graph.distanceMatrix[0]))
        agent = TransitionMatrixTree(graph, posTransitions, len(posTransitions),200, 200, 30, 0.7, 0.4, 0.9, 1.1,10)
        best = agent.mainRun()
        temp.append(best)
    print(temp)
    

    # convert array into dataframe 

    """ x = np.arange(200)  # Independent variable, common for both arrays
    y1 = agent.bestSolMatrix  # First dependent variable array
    y2 = agent.meanOverTime  # Second dependent variable array
    yerr = agent.sdOverTime  # Standard deviation for the second array
    yerr[0] = 0
    # Plotting the first array
    plt.plot(x, y1, label='Best found solution', color='blue')
    plt.plot(x, y2, label='Mean solution value for each iteration', color='red')

    # Plotting the second array with error bars
    #plt.errorbar(x, y2, yerr=yerr, label='Mean solution value with standard deviation for each iteration', color='red', fmt='o', linestyle='-', capsize=5)
    #plt.ylim(3323, 8000)
    # Adding labels and title
    plt.xlabel('Iteration number')
    plt.ylabel('solution quality')
    plt.legend()

    # Show the plot
    plt.show()
    ordered = np.column_stack((agent.offsetCorrelationTime, agent.genSolsTime, agent.batchCalcTime, agent.afterRemoveAndDeleteTime, agent.bestSolMatrix, agent.nRemoved, agent.nAdded, agent.meanOverTime, agent.sdOverTime))
    columnNames = ["offsetCorrelationTime" , "genSolsTime", "batchCalcTime", "afterRemoveAndDeleteTime", "bestSolMatrix", "nRemoved", "nAdded", "meanOverTime", "sdOverTime"]
    DF = pd.DataFrame(data =ordered, columns=columnNames) 
    # save the dataframe as a csv file 
    DF.to_csv("data1.csv") """
    return

if __name__ == "__main__":
    main()