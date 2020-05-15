# Introduction #

This is the repository for the model mentioned in the paper {ENTER PAPER NAME HERE}

# Data #
In order to replicate the results mentioned in the paper, first download the W\&I+LOCNESS Dataset from https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz .

We tested our system on the ABCN development set.

# Pre-Processing #

To run the pre-processing step, run the process.py file in the Pre-Processing folder. You will also need to pass in two arguments: the input data file and the output file you want to write your processed data to. We recommend writing the output to ./input_data/input.txt

        python process.py {PATH TO INPUT DATA (ABCN DEV SET)}/{NAME OF INPUT DATA FILE} ./input_data/input.txt

# Evaluation #

To evaluate each system, run

        python eval.py

This will print out the results every 10 times it runs the evaluation for each system.


