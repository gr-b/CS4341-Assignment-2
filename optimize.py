# File: optimize.py
# Griffin Bishop, David Deisde, Gianluca Tarquinio, Ian Vossoughi

import sys, random, math, operator
from random import randint
import copy
import time


def getFromFile(filename):
    file = open(filename,"r")
    nums = list(map(int, file.read().split()))
    file.close()
    return nums

# Randomly assign the numbers in the given list to buckets
# added deep copy so that original list is not deleted
def putInBins(numbers):
    bins = [[],[],[]]
    new_numbers = copy.deepcopy(numbers)
    i = 0
    while len(new_numbers) > 0:
        selection = random.randint(0,len(new_numbers)-1)
        #print( i%3, selection, len(numbers))
        bins[i % 3].append(new_numbers[selection])
        new_numbers.pop(selection)
        i += 1
    return bins

def printBins(bins):
    for i in range(len(bins)):
        print("Bin " + str(i+1) + ":", bins[i], "-->",
              str(eval("scoreBin" + str(i+1) + "(bins[i])")))

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

def isPrime(i):
    return i in [2, 3, 5, 7]

def scoreBin3(bin3):
    score = 0
    middle = int(len(bin3)/2)
    for i in range(middle):
        oldscore = score
        if isPrime(bin3[i]):
            score += 4
        elif i < 0:
            score += -2
        else:
            score += -bin3[i]
        #print("First half: " + str(i) + ":" + str(bin3[i]) + ":Score:" + str(score-oldscore))
    if middle % 2 == 0:
        middle -= 1
    for i in range(middle+1, len(bin3)):
        oldscore = score
        if isPrime(bin3[i]):
            score += -4
        elif bin3[i] < 0:
            score += 2
        else:
            score += bin3[i]
        #print("second half: " + str(i) + ":" + str(bin3[i]) + ":Score:" + str(score-oldscore))
    return score

def scoreBins(bins):
   return scoreBin1(bins[0]) + scoreBin2(bins[1]) + scoreBin3(bins[2])

def randomSelection(population):
    totalFitness = 0
    for org in population:
        totalFitness += org.score
    randScore = random.randrange(0, totalFitness)
    runningFitness = 0
    for org in population:
        if runningFitness > randScore:
            return org
        else:
            runningFitness += org.score


def breedOrganisms(population, newPopulation, popSize):
    while len(newPopulation) < popSize:
        parent1 = randomSelection(population)
        parent2 = randomSelection(population)
        while parent2 is parent1:
            parent2 = randomSelection(population)
        flatList1 = [y for x in parent1.bins for y in x]
        flatList2 = [y for x in parent2.bins for y in x]
        cutpoint = random.randrange(0, len(flatList1))

        child1List = flatList1[0:cutpoint] + flatList2[cutpoint:len(flatList2)]
        child2List = flatList2[0:cutpoint] + flatList1[cutpoint:len(flatList1)]
        oneThirdList = int(len(child1List) / 3)

        child1Bins = [child1List[0:oneThirdList],child1List[oneThirdList:2*oneThirdList],child1List[2*oneThirdList:3*oneThirdList]]
        child2Bins = [child2List[0:oneThirdList],child2List[oneThirdList:2*oneThirdList],child2List[2*oneThirdList:3*oneThirdList]]

        child1 = Organism(child1Bins, scoreBins(child1Bins))
        child2 = Organism(child2Bins, scoreBins(child2Bins))

        newPopulation.append(child1)
        newPopulation.append(child2)
    return newPopulation


def geneticAlgorithm(elite, popSize, nums, timeLimit):
    startTime = time.time()
    population = []
    elitism = math.ceiling(elite * popSize)
    for org in range(popSize):
        anOrg = Organism(putInBins(nums), 0)
        anOrg.score = scoreBins(anOrg.bins)
        population.append(anOrg)

    while (time.time() - startTime < timeLimit):
        newPopulation = []
        population.sort(key = operator.attrgetter('score'))
        i = 0
        while (i < elitism) and (i < len(population)):
            newPopulation[i] = population[i]
            i += 1
        population = breedOrganisms(population, newPopulation, popSize)



class Organism(object):
    def __init__(self, bins, score):
        self.bins = bins
        self.score = score
        
    def mutation(self, mutationProbability):
        """
        Mutation does random swaps between bins, similar to hill climbing

        :param mutation: the probability for the mutation
        :return: True/False if the mutation succeeded
        """
        ## if random variable is less than the mutationProbability, then grab a random number from a random bin
        ## if the random variable is not less, dont do anything

        # determine if the mutation can be moved
        random_probability = random.random()
        if not random_probability < mutationProbability:
            return False

        # now grab a random index in a random bin, and pick a random value from -9 to -9
        bin_length = len(self.bins[0])
        num_bins = 3

        random_bin = random.randint(1, 3)
        random_index = random.randint(0,bin_length-1)

        random_number_replacement = random.randint(-9, -9)

        bin_to_change = self.bins[random_bin]

        # add in the random number
        bin_to_change[random_index] = random_number_replacement

        return True


def getAllBinScores(bins):
    """
    Returns a tuple length of 3 with the scores of respective bins
    :param bins: The bins that the numbers are dropped in
    :return: Tuple() of scores for the 3 bins
    """
    return (scoreBin1(bins[0]), scoreBin2(bins[1]), scoreBin3(bins[2]))

def swap(bin1, idx1, bin2, idx2):
    temp = bin1[idx1]
    bin1[idx1] = bin2[idx2]
    bin2[idx2] = temp

def hillClimbing(numbers, timeLimit):
    #setup
    startTime = time.time()
    bestSolution = None
    length = len(numbers) / 3
    #keep searching for solution while there is time last
    while(time.time() - startTime < timeLimit):
        #randomly fill bins
        bins = putInBins(numbers)
        currentScore = scoreBins(bins)
        if(bestSolution == None):
            bestSolution = copy.deepcopy(bins)
        tries = 0
        while(tries < 100 and time.time() - startTime < timeLimit):
            #pick two random locations
            locations = [] #[first_bin, first_bin_index, second_bin, second_bin_index]
            locations.append(random.randrange(0, 3))
            locations.append(random.randrange(0, length))
            locations.append(random.randrange(0, 3))
            locations.append(random.randrange(0, length))
            while(time.time() - startTime < timeLimit and locations[0] == locations[2] and locations[1] == locations[3]):
                locations[2] = random.randrange(0, 3)
                locations[3] = random.randrange(0, length)

            #make a swap and get new score
            swap(bins[locations[0]], locations[1], bins[locations[2]], locations[3])
            score = scoreBins(bins)

            #check that move is an improvement
            if(score > currentScore):
                #if it is, reset tries, update currentScore, and update temperature
                tries = 0
                currentScore = score
            else:
                #if it isn't, swap it back and increment tries
                swap(bins[locations[0]], locations[1], bins[locations[2]], locations[3])
                tries += 1
        #if the new solution is better that the old best solution, replace the old best
        if(currentScore > scoreBins(bestSolution)):
            bestSolution = copy.deepcopy(bins)
    return bestSolution

def tryMove(newScore, oldScore, temperature):
    if(newScore > oldScore):
        return True
    else:
        prob = math.exp(float(newScore-oldScore) / temperature)
        return random.random() < prob;

def getTemp(time): #placeholder implementation
    return math.pow(0.95, time)

def simAnneal(numbers, timeLimit):
    #setup
    startTime = time.time()
    bestSolution = None
    length = len(numbers) / 3
    #keep searching for solution while there is time last
    while(time.time() - startTime < timeLimit):
        #randomly fill bins
        bins = putInBins(numbers)
        currentScore = scoreBins(bins)
        if(bestSolution == None):
            bestSolution = copy.deepcopy(bins)
        t = 1
        temperature = getTemp(t)
        if(t > 0):
            while(time.time() - startTime < timeLimit and temperature > 0): #this restart condition is subject to change
                #pick two random locations
                locations = [] #[first_bin, first_bin_index, second_bin, second_bin_index]
                locations.append(random.randrange(0, 3))
                locations.append(random.randrange(0, length))
                locations.append(random.randrange(0, 3))
                locations.append(random.randrange(0, length))
                while(time.time() - startTime < timeLimit and locations[0] == locations[2] and locations[1] == locations[3]):
                    locations[2] = random.randrange(0, 3)
                    locations[3] = random.randrange(0, length)

                #make a swap and get new score
                swap(bins[locations[0]], locations[1], bins[locations[2]], locations[3])
                score = scoreBins(bins)

                #try to make the move
                if(tryMove(score, currentScore, temperature)):
                    #if it works, update currentScore
                    currentScore = score
                else:
                    #if it fails, swap it back
                    swap(bins[locations[0]], locations[1], bins[locations[2]], locations[3])
                #Update temperature
                t += 1
                temperature = getTemp(t)
        #if the new solution is better that the old best solution, replace the old best
        if(currentScore > scoreBins(bestSolution)):
            bestSolution = copy.deepcopy(bins)
    return bestSolution



def getRandomBin(bins):
    """
    Gets a random bin
    :param bins:  the bins
    :return: returns a random bin index
    """
    bin_length = len(bins)
    index = randint(0, bin_length-1)

    return index

def getRandomNumInBin(bins):
    """
    Gets random number from the passed in bin
    :param bin: the bin chosen to get val from
    :return: random value within the bin
    """
    bin_size = len(bins)
    index = randint(0,bin_size-1)
    # val = bin[rand_int]
    return index


def main():
    arguments = sys.argv

    if len(arguments) != 4:
        print("Invalid Format, try: python optimize.py [hill, annealing, ga] [filename.txt] [seconds]")
        #exit()

    algorithm = arguments[1]
    filename = arguments[2]
    timelimit = float(arguments[3])

    nums = getFromFile(filename)

    bins = putInBins(nums)
    #printBins(bins)
    #print("Total: " + str(scoreBins(bins)))

    # hill climbing tests
    #best_solution = hillClimbing(bins, nums, time_limit=timelimit)
    #print "Caclualted again score %s. " % (sum(getAllBinScores(best_solution)))

    bestSolution = None
    if algorithm == "annealing":
        bestSolution = simAnneal(nums, timelimit)
    elif algorithm == "hill":
        bestSolution = hillClimbing(nums, timelimit)
    elif algorithm == "ga":
        bestSolution = geneticAlgorithm(0, 10, nums, timelimit)
    else:
        print("Incorrect algorithm name given")
        exit()

    #print(bestSolution)
    print("Score: " + str(scoreBins(bestSolution)))


if __name__ == '__main__':
    main()
