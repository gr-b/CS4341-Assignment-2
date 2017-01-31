# File: optimize.py
# Griffin Bishop, David Deisde, Gianluca Tarquinio, Ian Vossoughi

import sys, random

def getFromFile(filename):
    file = open(filename,"r")
    nums = list(map(int, file.read().split()))
    file.close()
    return nums

# Randomly assign the numbers in the given list to buckets
def putInBins(numbers):
    bins = [[],[],[]]
    i = 0
    while len(numbers) > 0:
        selection = random.randint(0,len(numbers)-1)
        #print( i%3, selection, len(numbers))
        bins[i % 3].append(numbers[selection])
        numbers.pop(selection)
        i += 1
    return bins

def printBins(bins):
    for i in range(len(bins)):
        print("Bin " + str(i) + ":", bins[i])

def scoreBins(bins):
    # First bin
    # Score: alternately add and subtract values
    score0 = 0
    i = 0
    for item in bins[0]:
        if i % 2:
            score0 += item
        else:
            score0 -= item
        i += 1

    score1 = 0
    # If value of i+1 > i, +3. if i+1==i, +5. if i+1 < i, -10
    for index in range(len(bins[1])-1):
        i = bins[1][index]
        iplus1 = bins[1][index+1]
        
        if  iplus1 > i:
            score1 += 3
        elif iplus1 == i:
            score1 += 5
        elif iplus1 < i:
            score1 -= 10

    score2 = 0
    #

    return (score0, score1, score2)

def main():
    arguments = sys.argv

    if len(arguments) != 4:
        print("Invalid Format, try: python optimize.py [hill, annealing, ga] [filename.txt] [seconds]")
        #exit()

    algorithm = arguments[1]
    filename = arguments[2]
    timelimit = float(arguments[3])

    nums = getFromFile(filename)
    #print(nums)
    bins = putInBins(nums)
    printBins(bins)


if __name__ == '__main__':
    main()
