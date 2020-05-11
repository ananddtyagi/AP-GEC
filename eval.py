from bert1 import predict as BERT1
# from bert2 import predict as BERT2
from electra1 import predict as ELECTRA1
# from electra2 import predict as ELECTRA2

from tqdm import trange
topk = 1
bert1_correct = 0
electra1_correct = 0
total = 0

with open('./input_data/input.txt', 'r') as preprocessed_file:
    lines = preprocessed_file.readlines()
    i = 0
    while i < len(lines):

        input_sentence = lines[i][2:] #exclude the beggining S
        correction_line = lines[i+1]

        i+=3#sets it to next set of sentences

        correction = correction_line.split('|||')

        if correction[1][0].lower() != 'r': #only want to look at replacements
            continue

        #skipping spelling errors as the model fails to identify them
        if correction[1][2:].lower() == 'spell':
            continue

        if len(correction[2].split(' ')) > 1: #can't handle multi replacements yet
            continue

        total += 1 #we will evalute this error

        bert1_suggestions = [sug['token'] for sug in BERT1(input_sentence, topk)]
        if correction[2] in bert1_suggestions:
            #NEED TO ADD CHECKING FOR THE SAME TOKEN INDEX BEING SWAPPED (USE THE INDEX PARAMETER)
            bert1_correct += 1

        electra1_suggestions = [sug['token'] for sug in ELECTRA1(input_sentence, topk)]
        if correction[2] in electra1_suggestions:
            #NEED TO ADD CHECKING FOR THE SAME TOKEN INDEX BEING SWAPPED (USE THE INDEX PARAMETER)
            electra1_correct += 1

        if total % 10 == 0:
            print('BERT1 results:')
            print("\t Correct: ", bert1_correct)
            print("\t Total Evaluated: ", total)
            print("\t Accuracy: ", bert1_correct/total)

            print('ELECTRA1 results:')
            print("\t Correct: ", electra1_correct)
            print("\t Total Evaluated: ", total)
            print("\t Accuracy: ", electra1_correct/total)

    print('BERT1 results:')
    print("\t Correct: ", bert1_correct)
    print("\t Total Evaluated: ", total)
    print("\t Accuracy: ", bert1_correct/total)

    print('ELECTRA1 results:')
    print("\t Correct: ", electra1_correct)
    print("\t Total Evaluated: ", total)
    print("\t Accuracy: ", electra1_correct/total)

