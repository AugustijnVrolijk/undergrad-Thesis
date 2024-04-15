
import numpy as np
import xml.etree.ElementTree as ET
import numpy as np


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

def fetchGraph(xmlFile:str) -> Graph:
    File = ET.parse(xmlFile)
    root = File.getroot()
    graph = root.find('graph')
    return Graph(graph)

def main():
    graph = fetchGraph('burma14.xml')
    
    test = np.arange(len(graph.distanceMatrix[0]))
    tester = np.column_stack((test, graph.distanceMatrix[0]))
    print(tester)
    tester = np.rint(tester[tester[:, 1].argsort(), 0]).astype(int)
    print(tester)
    bef = tester[0:6]
    aft = tester[6:]
    print(bef)
    print(aft)

if __name__ == "__main__":
    main()