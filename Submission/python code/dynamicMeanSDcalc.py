import numpy as np
import math 
import statistics

def calcMean(list):
    length = len(list)
    total = 0
    for i in list:
        total += i
    total = total/length
    return total


class stats():
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.bufferP = 0
        self.buffer = [0]*bufferSize
        self.squaredBuffer = [0]*bufferSize
        self.absBuffer = [0]*bufferSize
        self.mean = 0
        self.sd = 0
        self.variance = 0
        self.meanAbsoluteDeviation = 0
        self.fullBuffer = False
        self.itercount = 0
        self.S = 0
        return

    def PreBufferCalc(self, newVal):
        tempMean = self.mean
        
        meanTemp = (newVal - self.mean)/self.itercount
        self.mean += meanTemp

        temp = newVal - self.mean
        squaredTemp = (temp*temp)/self.bufferSize
        absTemp = abs(temp)/self.bufferSize

        self.S += (newVal-tempMean)*(newVal-self.mean)
        self.variance = self.S/self.itercount
        self.meanAbsoluteDeviation = ((self.meanAbsoluteDeviation*(self.itercount-1))+abs(newVal-self.mean))/self.itercount

        self.sd = math.sqrt(self.variance)

        self.squaredBuffer[self.bufferP] = squaredTemp
        self.absBuffer[self.bufferP] = absTemp
        self.buffer[self.bufferP] = newVal/self.bufferSize
        self.bufferP = self.bufferP + 1

        if self.itercount == self.bufferSize:
            self.fullBuffer = True
            self.bufferP = 0
            return
        return
    
    def MAD(self):
        totalSum = 0
        for i in range(len(self.buffer)):
            totalSum += abs(self.buffer[i] - self.mean)
            self.buffer[i] = self.buffer[i]/self.bufferSize
        self.meanAbsoluteDeviation = totalSum/self.bufferSize
        
    def addNumber(self, newVal):
        if self.fullBuffer:
            self.postBufferCalc(newVal)
        else: 
            self.itercount += 1
            self.PreBufferCalc(newVal)

    def postBufferCalc(self, newVal):
        temp = newVal - self.mean
        squaredTemp = (temp*temp)/self.bufferSize
        absTemp = abs(temp)/self.bufferSize
        meanTemp = newVal/self.bufferSize

        self.variance += squaredTemp - self.squaredBuffer[self.bufferP]
        self.meanAbsoluteDeviation += absTemp - self.absBuffer[self.bufferP]
        self.mean += meanTemp - self.buffer[self.bufferP]

        self.sd = math.sqrt(self.variance)

        self.squaredBuffer[self.bufferP] = squaredTemp
        self.absBuffer[self.bufferP] = absTemp
        self.buffer[self.bufferP] = meanTemp
        self.bufferP = (self.bufferP + 1)%self.bufferSize

    

def main():

    numbers = np.random.normal(100, 15, 300)
    print(numbers)
    trueMean = calcMean(numbers)
    print("true mean is: {}".format(trueMean))
    print("true stdev is: {}".format(statistics.stdev(numbers)))

    temp = stats(100)
    for i in range(len(numbers)):
        if i%25 == 0:
            print("\niteration is: {}".format(i))
            print("calculated mean is: {}".format(temp.mean))
            print("calculated MDS is: {}".format(temp.meanAbsoluteDeviation))
            print("calculated variance is: {}".format(temp.variance))
            print("calculated SD is: {}".format(temp.sd))


        temp.addNumber(numbers[i])
    
    print("calculated mean is: {}".format(temp.mean))
    print("calculated MDS is: {}".format(temp.meanAbsoluteDeviation))
    print("calculated variance is: {}".format(temp.variance))
    print("calculated SD is: {}".format(temp.sd))

    return

if __name__ == "__main__":
	main()