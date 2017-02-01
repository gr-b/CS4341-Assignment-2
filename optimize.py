# File: optimize.py
# Griffin Bishop, David Deisde, Gianluca Tarquinio, Ian Vossoughi

import sys, random
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
   #print("First bin: " + str(scoreBin1(bins[0])))
   #print("Second bin: " + str(scoreBin2(bins[1])))
   #print("Third bin: " + str(scoreBin3(bins[2])))
   return scoreBin1(bins[0]) + scoreBin2(bins[1]) + scoreBin3(bins[2])

def getAllBinScores(bins):
    """
    Returns a tuple length of 3 with the scores of respective bins
    :param bins: The bins that the numbers are dropped in
    :return: Tuple() of scores for the 3 bins
    """
    return (scoreBin1(bins[0]), scoreBin2(bins[1]), scoreBin3(bins[2]))

def moveOperator():
    """
    Determines all possible moves for a given solution
    :return:
    """

def hillClimbing(bins, numbers, time_limit, max_tries=100):
    """
    Performs hill climbing to find best scored bins for the given numbers.

    :param numbers: Input numbers passed in
    :param time_limit: time limit allowed for the function
    :param max_tries: maximum moves allowed
    :return: returns the updated bins, if any
    """
    # record the start time for the function
    start_time = time.time()

    # Bin Swapping logic: get random bin, and random value from that bin. Same for other bin, if their swap beats the best score, then make the swap

    best_score_individual = getAllBinScores(bins)
    best_score = sum(best_score_individual)
    best_solution_bins = copy.deepcopy(bins)

    num_bins = len(bins)

    has_time = True
    while has_time:
        # reset the local score and reset the bins
        tries = 0
        best_score_local = best_score
        bins = putInBins(numbers)

        while tries < max_tries:
            # do we want to do states that we have already seen?
            # already_seen_move = []
            # new_bins = bins.pop(cur_i)

            # get the two random bin indices
            starting_bin_ind = getRandomBin(bins)
            other_bin_ind = getRandomBin(bins)

            # get the two random bin value indices
            starting_bin_val_ind = getRandomNumInBin(bins)
            other_bin_val_ind = getRandomNumInBin(bins)

            # check to see if the same value is not selected in the same bin
            while other_bin_ind == starting_bin_ind and starting_bin_val_ind == other_bin_val_ind:
                other_bin_val_ind = getRandomNumInBin(bins)

            # make swap here
            bins[starting_bin_ind][starting_bin_val_ind] = bins[other_bin_ind][other_bin_val_ind]
            tries += 1

            # get the current score after the swap
            score = sum(getAllBinScores(bins))
            best_score_global = sum(getAllBinScores(best_solution_bins))

            # check the score against the best score, if the swap was successfull, finalize the change, otherwise swap back
            if score > best_score_global:
                best_score = score
                best_solution_bins = copy.deepcopy(bins)
            else:
                # swap them back
                bins[starting_bin_ind][starting_bin_val_ind] = bins[other_bin_ind][other_bin_val_ind]

            # check time here
            end_time = time.time()
            delta_time = end_time - start_time

            if delta_time >= time_limit:
                print "time ended"
                print "Best Score: ", best_score_global, best_solution_bins
                print "Current score:", score, bins

                has_time = False
                break


    return best_solution_bins

def swap(bin1, idx1, bin2, idx2):
    temp = bin1[idx1]
    bin1[idx1] = bin2[idx2]
    bin2[idx2] = temp
    
def tryMove(newScore, oldScore, temperature): #placeholder implementation
    return newScore > oldScore
    
def getNextTemp(temperature): #placeholder implementation
    return temperature

def simAnneal(numbers, timeLimit, startTemp):
    #setup
    startTime = time.time()
    bestSolution = None
    length = len(numbers) / 3
    temperature = startTemp
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
            
            #try to make the move
            if(tryMove(score, currentScore, temperature)):
                #if it works, reset tries, update currentScore, and update temperature
                tries = 0
                currentScore = score
                temperature = getNextTemp(temperature)
            else:
                #if it fails, swap it back and increment tries
                swap(bins[locations[0]], locations[1], bins[locations[2]], locations[3])
                tries += 1
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
    
    bestSolution = simAnneal(nums, timelimit, 17)
    #bestSolution = hillClimbing(bins, nums, time_limit=timelimit)
    print("Score: " + str(scoreBins(bestSolution)))
    print(bestSolution)


if __name__ == '__main__':
    main()
