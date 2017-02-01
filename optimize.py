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

def scoreBin1(bin1):
     # First bin
    # Score: alternately add and subtract values
    score = 0
    i = 0
    for item in bin1:
        if not i % 2:
            score += item
        else:
            score -= item
        i += 1
    return score

def scoreBin2(bin2):
    score = 0
    # If value of i+1 > i, +3. if i+1==i, +5. if i+1 < i, -10
    for index in range(len(bin2)-1):
        i = bin2[index]
        iplus1 = bin2[index+1]
        
        if  iplus1 > i:
            score += 3
        elif iplus1 == i:
            score += 5
        elif iplus1 < i:
            score -= 10
    return score

def scoreBins(bins):
   print("First bin: " + str(scoreBin1(bins[0])))
   print("Second bin: " + str(scoreBin2(bins[1])))

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
    scoreBins(bins)

if __name__ == '__main__':
    main()
