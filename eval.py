# from bert1 import predict as BERT1
# from bert2 import predict as BERT2
# from electra1 import predict as ELECTRA1
# from electra2 import predict as ELECTRA2
from bert import predict as BERT
from electra_base import predict as ELECTRA_BASE
from electra_combined import predict as ELECTRA_COMBINED
from combined_both import predict as COMBINED_BOTH

from tqdm import trange

bert_correct = 0
electra_base_correct = 0
electra_combined_correct = 0
combined_both = 0

total = 0
threshold = 0.8

with open('./input_data/test.txt', 'r') as preprocessed_file:
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

        bert_suggestions = BERT(input_sentence, threshold)
        if correction[2] in bert_suggestions:
            bert_correct += 1

        electra_base_suggestions = ELECTRA_BASE(input_sentence, threshold)
        if correction[2] in electra_base_suggestions:
            electra_base_correct += 1

        electra_combined_suggestions = ELECTRA_COMBINED(input_sentence, threshold)
        if correction[2] in electra_combined_suggestions:
            electra_combined_correct += 1

        combined_both_suggestions = COMBINED_BOTH(input_sentence, threshold)
        if correction[2] in combined_both_suggestions:
            combined_both_correct += 1

        if total % 10 == 0:
            print("\nTotal Evaluated: ", total)

            print('BERT results:')
            print("\t Correct: ", bert_correct)
            print("\t Accuracy: ", bert_correct/total)

            print('Electra Base results:')
            print("\t Correct: ", electra_base_correct)
            print("\t Accuracy: ", electra_base_correct/total)

            print('Combined Electra results:')
            print("\t Correct: ", electra_combined_correct)
            print("\t Accuracy: ", electra_combined_correct/total)

            print('Combined Both results:')
            print("\t Correct: ", combined_both_correct)
            print("\t Accuracy: ", combined_both_correct/total)


    print("\nTotal Evaluated: ", total)

    print("\nTotal Evaluated: ", total)

    print('BERT results:')
    print("\t Correct: ", bert_correct)
    print("\t Accuracy: ", bert_correct/total)

    print('Electra Base results:')
    print("\t Correct: ", electra_base_correct)
    print("\t Accuracy: ", electra_base_correct/total)

    print('Combined Electra results:')
    print("\t Correct: ", electra_combined_correct)
    print("\t Accuracy: ", electra_combined_correct/total)

    print('Combined Both results:')
    print("\t Correct: ", combined_both_correct)
    print("\t Accuracy: ", combined_both_correct/total)
