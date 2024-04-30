import numpy as np

def calcVectorLength(directionVector:list) -> int:
    temp = np.square(directionVector) #temp is a list, where each element is the squared value of directionvector
    length = np.sum(temp)
    return np.sqrt(length)

def calcDirectionalDeriv(directionVector:list, currPointVal: float, prevPointVal: float) -> float:
    vLength = calcVectorLength(directionVector)
    
    deriv = (currPointVal - prevPointVal)/vLength
    return deriv

def normaliseVector(directionVector: np.array, vLength:float = None) -> list:
    if vLength == None:
        vLength = calcVectorLength(directionVector)
    normalVector = np.empty(directionVector.size)
    np.divide(directionVector, vLength, normalVector)
    return normalVector


def main():
    test = np.array([1,5,3,2,5])
    temp = normaliseVector(test)
    print(temp)
    print(calcVectorLength(temp))
    return

if __name__ == "__main__":
    main()