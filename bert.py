#This code was written by Anand Tyagi

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys


# setup toolbar
#ELECTRA
# from transformers import ElectraModel, ElectraTokenizer, ElectraForMaskedLM, ElectraForTokenClassification

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)

def bert_format(sentence, masked_index):
    return ['[CLS]'] + sentence + ['[SEP]'], masked_index + 1

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


train_file = open('./Pre-Processing/output.txt')

sentence = ''
edit = ''
k = 1 #for how many top predictions
same_count = 0
replace_count = 0
stop = 1
stop_counter = 0
tokenized_sentence = None
masked_index = None
indexed_tokens = None
segments_ids = None
tokens_tensor = None
segments_tensors = None

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# model = ElectraForMaskedLM.from_pretrained('google/electra-smaqll-discriminator')

model.eval()
model.to('cuda')
output = None
prediction = None
topk_values = None
topk_indicies = None
predicted_tokens = None

for line in train_file:

    # if stop_counter == stop:
    #     break;
    # if stop_counter % 10 == 0:
    #     print(stop_counter / stop, ' % done')
    sys.stdout.flush()

    sys.stdout.write("\r%i done" % (stop_counter * 100 / 87287))


    if line[0] == 'S':
        sentence = ''
        edit = ''
        sentence = line[2:]
    elif line[0] == 'A':
        edit = line
    elif line[0] == '\n':
        edit = edit.split('|||')
        stop_counter += 1
        if stop_counter % 1000 == 0:
            print(stop_counter, " tests  ", replace_count, " = total checked  ", same_count, "=same count   % = ", same_count / replace_count)

        if edit[1][0] == "R": #only if it's an R HANDLE LATER
            if edit[1][2:] == 'SPELL': #HANDLE LATER
                continue;

            if len(edit[2].split(' ')) > 1: #HANDLE LATER, split each edit in data, even if it's a single replacement, to one word edit per total edit
                continue;

            if int(edit[0].split(' ')[2]) -  int(edit[0].split(' ')[1]) != 1: #HANDLE LATER, can't handle the replacement of multiple words
                continue;


            #
            tokenized_sentence = sentence.strip().split(' ')

            masked_index = int(edit[0].split(' ')[1])
            tokenized_sentence[masked_index] = '[MASK]'
            tokenized_sentence = ' '.join(tokenized_sentence)
            tokenized_sentence = tokenizer.tokenize(tokenized_sentence)
            masked_index = tokenized_sentence.index('[MASK]')
            tokenized_sentence, masked_index = bert_format(tokenized_sentence, masked_index)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)

            segments_ids = [0] * len(tokenized_sentence) #since there is only one mask token and the [SEP] is only at the end

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # put everything on GPUs
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')


            # Predict all tokens
            with torch.no_grad():
                outputs = model(tokens_tensor, token_type_ids=segments_tensors)
                predictions = outputs[0] #logits

            # topk predictions
            topk_values, topk_indicies = torch.topk(predictions[0, masked_index], k)
            predicted_tokens = tokenizer.convert_ids_to_tokens([topk_indicies[0]])


            if predicted_tokens[0].lower() == edit[2].lower():
                same_count += 1
            # else:
                # print(tokenized_sentence)
                # print(masked_index)
                # print(predicted_tokens[0])
                # print(edit[2])
                # print('\n')
            replace_count += 1


print("percent replacement same = ", same_count / replace_count)

# Tokenize input
#
# tokenized_text = tokenizer.tokenize(text)
#
# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 5
# tokenized_text[masked_index] = '[MASK]'
# tokenized_text = ['[CLS]', 'I', 'like', 'to', 'take', '[MASK]', 'in', 'the', 'park', '.','[SEP]']

# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# # Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])
#
# # Load pre-trained model (weights)
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# model.eval()

# # If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')
#
# # Predict all tokens
# with torch.no_grad():
#     outputs = model(tokens_tensor, token_type_ids=segments_tensors)
#     predictions = outputs[0] #logits



