
import xml.etree.ElementTree as ET
import numpy as np
import math
from operator import itemgetter
import matplotlib.pyplot as plt

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

class agent():
	def __init__(self, chromosome = None, inverse = None):
		self.score = None
		if chromosome is None:
			self.inverse = inverse
			self.calcChromosome()
		else:
			self.chromosome = chromosome
			self.calcInverse()

	def flipMutation(self) -> None:
		pos1 = np.random.randint(len(self.chromosome))
		pos2 = np.random.randint(len(self.chromosome))
		while pos1 == pos2:
			pos2 = np.random.randint(len(self.chromosome))
		
		temp = self.chromosome[pos1]
		self.chromosome[pos1] = self.chromosome[pos2]
		self.chromosome[pos2] = temp
		self.calcInverse()
		return 
	
	def calcChromosome(self) -> None:
		length = len(self.inverse)
		self.chromosome = np.empty(length, dtype=int)
		pos = np.zeros(length, dtype=int)
		for i in range(length-1, -1, -1):
			for j in range(i, length):
				if pos[j] >= self.inverse[i]:
					pos[j] += 1
			pos[i] = self.inverse[i]

		for i in range(length):
			self.chromosome[pos[i]] = i
		return 
	
	def calcInverse(self) -> None:
		length = len(self.chromosome)
		self.inverse = np.zeros(length, dtype=int)
		for i in range(length):
			m = 0
			while self.chromosome[m] != i:
				if self.chromosome[m] > i:
					self.inverse[i] += 1
				m += 1 
		return

	def calcScore(self, graph:Graph) -> None:
		self.score = graph.calcDistance(self.chromosome)
		return

# tournament selection
"""
input: pop -> sorted list of the current population by their score
	   k -> number of individuals to select for tournament
"""
def TournamentSelection(pop:list, k:int)-> list[tuple]:
	
	if k < 2:
		print("tournament size too small")
		exit()

	finalPop = np.empty(int(len(pop)/2), dtype=object)

	for i in range(int(len(pop)/2)):
		temp = np.empty(k, dtype=int)

		for j in range(k):
			temp[j] = np.random.randint(len(pop))

		temp = np.sort(temp, 0)
		finalPop[i] = (pop[temp[0]], pop[temp[1]])

	return finalPop

def invSeqCrossover(agent1:agent, agent2:agent) -> tuple:
	crossoverPoint = np.random.randint(1, len(agent1.inverse)-1)
	temp1 = np.empty(len(agent1.inverse), dtype=int)
	temp2 = np.empty(len(agent1.inverse), dtype=int)

	for i in range(len(agent1.inverse)):
		if i < crossoverPoint:
			temp1[i] = agent1.inverse[i]
			temp2[i] = agent2.inverse[i]
		else:
			temp1[i] = agent2.inverse[i]
			temp2[i] = agent1.inverse[i]
	
	child1 = agent(inverse=temp1)
	child2 = agent(inverse=temp2)

	return child1, child2

def GA(popsize:int, cityNumber:int, graph:Graph, iterCount:int, tournamentSize:int, mutationProb:float) -> agent:
	bestSolMatrix = np.empty(iterCount,dtype=float)
	meanOverTime = np.empty(iterCount,dtype=float)
	#generate initial pop
	pop = [None] * popsize
	for i in range(popsize):
		temp = genRandomChromosome(cityNumber)
		pop[i] = agent(chromosome=temp)
	#initialise best iteration
	bestAgent = pop[0]

	for i in range(iterCount):
		for Agent in pop:
			Agent.calcScore(graph)
		pop.sort(key=lambda x: x.score)
		if pop[0].score < bestAgent.score:
			bestAgent = pop[0]
			print(">%d, new best f(%s) = %.3f" % (i,  bestAgent.chromosome, bestAgent.score))

		newpop = TournamentSelection(pop, tournamentSize)
		iter = 0
		mutate = np.random.uniform(0,1,popsize)
		for parents in newpop:
			child1, child2 = invSeqCrossover(parents[0], parents[1])
			if mutate[iter] <= mutationProb:
				child1.flipMutation()
			if mutate[iter+1] <= mutationProb:
				child2.flipMutation()

			pop[iter] = child1
			pop[iter+1] = child2
			iter += 2
		

	for Agent in pop:
			Agent.calcScore(graph)
	return bestAgent, pop

def genRandomChromosome(length):
	array = np.arange(length, dtype=int)
	np.random.shuffle(array)
	return array

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)

def main():
	
	graph = fetchGraph('burma14.xml')
	temp = []
	for i in range(10):
		best, pop = GA(50,len(graph.distanceMatrix[0]), graph,50,10,0.05)
		print("best solution score: {}, solution: {}".format(best.score, best.chromosome))
		temp.append(best.score)
	print(temp)
	return

if __name__ == "__main__":
	main()