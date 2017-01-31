# File: optimize.py
# Griffin Bishop, David Deisde, Gianluca Tarquinio, Ian Vossoughi

import sys, random

def getFromFile(filename):
    file = open(filename,"r")
    nums = list(map(int, file.read().split()))
    file.close()
    return nums

# Randomly assign the numbers in the given list to buckets
def putInBuckets(numbers):
    buckets = [[],[],[]]
    i = 0
    while len(numbers) > 0:
        selection = random.randint(0,len(numbers)-1)
        #print( i%3, selection, len(numbers))
        buckets[i % 3].append(numbers[selection])
        numbers.pop(selection)
        i += 1
    return buckets

def printBuckets(buckets):
    for i in range(len(buckets)):
        print("Bucket " + str(i) + ":", buckets[i])
    
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
    buckets = putInBuckets(nums)
    printBuckets(buckets)


if __name__ == '__main__':
    main()
