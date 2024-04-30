
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
    def __init__(self, depth, momentum, transitionMatrix, isRoot):
        self.branches = []
        self.expired = []
        self.momentum = momentum
        self.depth = depth
        self.prob = 0
        self.transitionProb = 0
        self.beenExplored = 0
        self.transitionMatrix = transitionMatrix
        self.validBranches = 0
        self.isRoot = isRoot
        


    def addBranch(self, branch, maxTreeDepth) -> None:
        if self.depth == maxTreeDepth:
            return
        
        if self.branchExists(branch):
            return

        childMomentum = self.momentum
        if self.momentum < 0.5:
            childMomentum = 0.5
        newBranch = TreeNode(self, branch, childMomentum, False)
        self.branches.append(newBranch)
        self.validBranches += 1
        return
    
    def branchExists(self, parent):
        for branch in self.expired:
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
        total = 0
        
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
        self.transitionProb = TransitionProb**(2-self.momentum)
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
    
    def updateMomentum(self, isNewBest, MomentumPen):
        if isNewBest:
            self.momentum = 2
        else:
            val = self.momentum - MomentumPen
            if val > 0:
                self.momentum = val
            else:
                self.momentum = 0
                if not self.isRoot:
                    self.parent.expireBranch(self)

    def expireBranch(self, branch):
        try:
            self.branches.remove(branch)
        except:
            return
        self.expired.append(branch)
        self.validBranches -= 1

class TreeNode(Node):
    def __init__(self, parent, transition, momentum, isRoot) -> None:
        super().__init__(depth = parent.depth + 1, momentum=momentum, transitionMatrix=parent.transitionMatrix,isRoot=isRoot)
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

        if self.validBranches == 0:
            return path, self

        path, key = super().chooseTSPTransition(solution, validSols, path)
        return path, key
        
    def updateSelf(self, score, corrVal, MomentumPen = 0.02):
        isNewBest = self.transition.updateTransition(score, corrVal, MomentumPen)
        if isNewBest:

            self.transition.updateMomentum(isNewBest, MomentumPen)
        super().updateMomentum(isNewBest, MomentumPen)
        self.parent.updateTransition(score, corrVal, MomentumPen)
        return
    
    def updateTransition(self, score, corrVal, MomentumPen):

        isNewBest = self.transition.updateTransition(score, corrVal, MomentumPen)
        if isNewBest:
            super().updateMomentum(isNewBest, MomentumPen)
            self.transition.updateMomentum(isNewBest, MomentumPen)
        self.parent.updateTransition(score, corrVal, MomentumPen)
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
    
class Transition(Node):
    def __init__(self, val, dimension, transitionMatrix) -> None:
        super().__init__(depth=1, momentum=1, transitionMatrix=transitionMatrix,isRoot = True)
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

        if self.validBranches == 0:
            return path, self
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
    
    def updateSelf(self, score, corrVal, MomentumPen = 0.02):
        isNewBest = self.updateTransition(score, corrVal, MomentumPen)
        super().updateMomentum(isNewBest=isNewBest, MomentumPen=MomentumPen)
        return
    
    def updateTransition(self, score, corrVal, MomentumPen):
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

    def removeChild(self, child):
        self.branches.remove(child)

    def delete(self):
        print("we are deleting root with val:", self.best)
        for branch in self.branches:
            branch.delete()
        #momentum set to 0 means it will never be called again
        self.momentum = 0
        return
    
    def getRoot(self):
        return self

    def addBranch(self, branch, maxTreeDepth):
        super().addBranch(branch, maxTreeDepth)
        return
    
    def printSelf(self, noisy=True):
        if noisy:
            return print("node: (dim:{},val:{}), has best: {}, corr: {}, momen: {}".format(self.dimension, self.val, self.best, self.correlation, self.momentum))
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

    def __init__(self, optimisationFunc, posTransitions, Ndimen,iterCount:int, popSize:int, maxTreeDepth, alpha, beta, nAddBranch, offSet, corrRed):
        
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
        self.bestSol = None
        self.ratio = 0
        self.prevVal = np.inf
        self.prevBest = []
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1
        self.maxTreeDepth = maxTreeDepth
        self.nAddBranch = nAddBranch
        self.offSet = offSet
        self.corrRed = corrRed

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
            #add, rm = self.heuristicRankSols(solutions, sdRatio, 2)
            self.addEdgesTSP(add)
            #self.removeEdges(rm)
            
            afterRemoveAndAdd =  time.perf_counter()
            rmAndAdd = afterRemoveAndAdd - afterBatchCalc
            self.offSetCorr()
            
            self.printMomentum()
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

        return
    
    def printMomentum(self):
        print('\n'.join([''.join(['({:.2f}, {:.2f})'.format(self.transitionMatrix[row][col].momentum, self.probMatrix[row][col]) for row in range(self.Ndimen)]) for col in range(self.Ndimen)]))
        """ temp = []
        for i in range(self.Ndimen):
            #
            for j in range(self.Ndimen):

                totalMomentum += self.transitionMatrix[i][j].momentum

            temp.append((i, totalMomentum))
        print(temp) """

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
    
    def rankSolutions(self, solutions):
        temp = np.argsort(solutions[:, 0])
        add = solutions[temp[0:self.nAddBranch], 1]
        rm = solutions[temp[(self.popSize-self.nAddBranch):], :]
        return add, rm

    def heuristicRankSols(self, solutions, addHeuristic, rmHeuristic):
        add = []
        rm = []
        for sol in solutions:
            #alternatively self.mean - addheurstic*self.sd
            if sol[0] < self.best + addHeuristic*self.sd:
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
            #reward = self.calcQlearningReward(sol)
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
        lr = 0.9 #learning rate
        dr = 0.7 #discount rate
        reward =  lr*(-score + dr*(self.getBestNext(edge.val))- edge.correlation)     
        return reward

    def getBestNext(self, dimension):
        best = self.transitionMatrix[dimension, 0].correlation
        for edge in self.transitionMatrix[dimension, 1:]:
            if edge.correlation > best:
                best = edge.correlation
        return best

    def updateGaussianCorrelation(self, solutionID, score):
        #solutionID format: [(vertexID, edgeID),...,...]
        reward = self.calcGaussianReward(score)
        for edge in solutionID:    
            edge.updateSelf(score, reward, MomentumPen = 0.005)
    
    def updateQLearningCorrelation(self, solutionID, score):
        #solutionID format: [(vertexID, edgeID),...,...]
        for edge in solutionID:
            reward = self.calcQlearningReward(score, edge,0.9, 0.9)
            edge.updateSelf(score, reward)

    def genTSPSolution(self, random= False):

        solutionID = []
        validSols = np.arange(self.Ndimen).tolist()
        curDim = np.random.randint(0,self.Ndimen)
        validSols.remove(curDim)
        solution = [curDim]
        visited = 1

        while visited < self.Ndimen:
            
            if not random:
                total = np.sum(self.probMatrix[curDim,validSols])
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
    
    graph = fetchGraph('brazil58.xml')
    #iter count, popsize, correlation decay, offset: >1
    print("got graph")
    posTransitions = np.arange(len(graph.distanceMatrix[0]))
    agent = TransitionMatrixTree(graph, posTransitions, len(posTransitions), 700, 500, 30, 1, 0.3, 20, 0.9, 1.1)
    print("got agent")

    agent.mainRun()
    # convert array into dataframe 
    x = np.arange(700)  # Independent variable, common for both arrays
    y1 = agent.bestSolMatrix  # First dependent variable array
    y2 = agent.meanOverTime  # Second dependent variable array
    yerr = agent.sdOverTime  # Standard deviation for the second array
    yerr[0] = 0
    # Plotting the first array
    plt.plot(x, y1, label='Best found solution', color='blue')

    # Plotting the second array with error bars
    plt.plot(x, y2, label='Mean solution value each iteration', color='red')
    #plt.errorbar(x, y2, yerr=yerr, label='Mean solution value with standard deviation for each iteration', color='red', fmt='o', linestyle='-', capsize=1)
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
    DF.to_csv("data1.csv")
    return

if __name__ == "__main__":
    main()