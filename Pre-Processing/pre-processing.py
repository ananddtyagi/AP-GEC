
#import functions from independent files to keep each process separate
# from phase1 import phase1 #ignore for baseline
from phase2 import phase2
from phase3 import phase3 #takes one array [input file path, output file path]
############################



train_file = open('../wi+locness/m2/A.train.gold.bea19.m2', 'r')

sentence = ''
corrections = []

for line in train_file:

    if line[0] == '\n':
        phase1(sentence, corrections)
        sentence = ''
        corrections = []
        break;
    if line[0] == 'S':
        sentence = line
    if line[0] == 'A':
        corrections.append(line)

