Griffin Bishop
David Deisde
Gianluca Tarquinio
Ian Vossoughi

CS4341: Assignment 2 - Hill Climbing, Annealing, Genetic Algorithm Optimization

This program, written in Python, can read in a series of up to 9999 numbers, ranging from -9 to 9, and sort them into bins in order to maximize the score according to the specified rules:
Bin #1:  the scoring function is to alternately add and subtract the values in the bin.
Bin #2:  for every pair of numbers at positions i and i+1, if the value at position i+1 is larger than position i, it scores +3.  If the value at position i+1 is equal to the value at position i, it scores +5.  If the value at position i+1 is smaller than at position i, it scores -10.
Bin #3:  (positive) prime numbers in the first half of this bin score +4, all negatives values score -2, while composite positives score negative whatever their value is.  For the second half of the bin, these values are inverted:  primes scores -4, negatives score +2, and composite positives score whatever their value is.  In the event of an odd number length, the value exactly in the middle is ignored.  For purposes of bin #3, 0 and 1 are both composite numbers.
The type of search algorithm (hill climbing, annealing, or GA) can be specified in the command line.  

To run this program, unzip the Zip folder, navigate to this folder using the terminal,
and use the following command line command:
> python optimize.py [type of search] [text file of numbers] [number of seconds]

For instance, to run the file “tune.txt” using genetic algorithms for 3 seconds, you would type:
python optimize.py ga tune.txt 3

The search types are named “hill”, “annealing”, and “ga” for this program.

The program will output the high solution score discovered and the time required to obtain this value.