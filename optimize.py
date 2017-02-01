# File: optimize.py
# Griffin Bishop, David Deisde, Gianluca Tarquinio, Ian Vossoughi

import sys, random
from random import randint

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
        print("Bin " + str(i+1) + ":", bins[i])

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
   print("First bin: " + str(scoreBin1(bins[0])))
   print("Second bin: " + str(scoreBin2(bins[1])))
   print("Third bin: " + str(scoreBin3(bins[2])))

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

def hillClimbing(bins, max_tries=100):
    """
    Performs hill climbing to find best scored bins for the given numbers.

    :param numbers: Input numbers passed in
    :return: None
    """
    # Bin Swapping logic: get random bin, and random value from that bin. Same for other bin, if their swap beats the best score, then make the swap

    best_score_individual = getAllBinScores(bins)
    best_score = sum(best_score_individual)

    num_bins = len(bins)
    tries = 0

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

        # check the score against the best score, if the swap was successfull, finalize the change, otherwise swap back
        if score > best_score:
            best_score = score
            continue
        else:
            # swap them back
            bins[starting_bin_ind][starting_bin_val_ind] = bins[other_bin_ind][other_bin_val_ind]

    return bins


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
    #print(nums)
    bins = putInBins(nums)
    printBins(bins)
    scoreBins(bins)

    # hill climbing tests
    hillClimbing(bins)

if __name__ == '__main__':
    main()
