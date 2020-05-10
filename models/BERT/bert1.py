import torch
from transformers import pipeline, BertTokenizer, BertModel, BertForMaskedLM
import sys, string
import logging
logging.basicConfig(level=logging.INFO)
from operator import itemgetter

def fstr(sentence):
    return eval(f"f'{sentence}'")

def mask(sentence, masked_index):
    return sentence[:masked_index] + ['[MASK]'] + sentence[masked_index+1:]

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased",
    tokenizer="bert-base-uncased"
    )

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sent1 = 'HuggingFace is creating an tool that the community uses to solve NLP tasks.'
sent2 = "My town is a medium size city with eighty thousand inhabitants ."
sent3 = "It has a high - density population because its small territory ."
sent4 = "I recommend visiting the artificial lake in the certer of the city which is surrounded by a park ."
sent5 = "The quality of the products and services is quite good , because there is huge competition . However recommend you be careful about some fakes or cheats ."

data = {'sentence': sent5, 'correction': '', 'index': []}
tokenized_sentence = tokenizer.tokenize(data['sentence'])

top_k_pred = 5

def mask_predict_token(tokenized_sentence):
    best_predictions = []
    orignal_sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_sentence))

    for i, word in enumerate(tokenized_sentence):
        if tokenized_sentence[i] in string.punctuation: #avoid calculating punctuation which will likely have the highest result
            continue;

        masked_sentence = tokenizer.convert_tokens_to_string(mask(tokenized_sentence, i)).replace(' ##', '')

        #Note for later: make sure these partial tokens (##) won't cause any problems

        #ranked 1 prediction
        prediction = fill_mask(masked_sentence)[0]

        #removing special tokens
        tokenized_prediction = tokenizer.tokenize(prediction['sequence']) #tokenizing prediction sentence
        no_special_tokens_sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_prediction), skip_special_tokens=True) #actual sentence

        if (orignal_sentence != no_special_tokens_sentence):
            prediction['no_special_token_sentence'] = no_special_tokens_sentence
            best_predictions.append(prediction)

        #print(word, prediction)

    sorted_best_predictions = sorted(best_predictions, key=itemgetter('score'), reverse=True)

    highest_confidence_corrections = []
    for k in range(min(top_k_pred, len(orignal_sentence))):
        highest_confidence_corrections.append(sorted_best_predictions[k])

    top_k_tokens = []
    for ele in highest_confidence_corrections:
        top_k_tokens.append(tokenizer.convert_ids_to_tokens(ele['token']))

    # for ele in sorted_best_predictions:
    #     print(ele)

    #toke sentence with highest confidence correction without special tokens
    return(top_k_tokens)

print(mask_predict_token(tokenized_sentence))

preprocessed_file = open('./output3.txt')
preprocessed_file_str = preprocessed_file.read()

preprocessed_file_split = preprocessed_file_str.split('\n\n')
sentence_pairs = [ele.split('\n') for ele in preprocessed_file_split]

#remove the 'S' part of the sentences, and extract only the word the edits. Also, only keep replacement edits.
new_sentence_pairs = []
for ele in sentence_pairs:
    try:
        temp0 = ele[0][2:]
        temp1 = ele[1].split("|||")
    except IndexError:
        continue

    if (temp1[1][0].lower() != 'r'):
        continue

    #skipping spelling errors as the model fails to identify them
    if (temp1[1][2:].lower() == 'spell'):
        continue

    temp1 = temp1[2].lower()
    new_sentence_pairs.append([temp0, temp1])

#now applying it on the new sentence pairs
from tqdm import tqdm
import string

table = str.maketrans('', '', string.punctuation)

count = 0
correct = 0

N = 15
i = 0

#pbar = tqdm(total=len(new_sentence_pairs))
for pair in new_sentence_pairs:
    sentence = pair[0]

    correct_token = pair[1]
    #removing punctuation because BERT predictions don't have punctuation
    correct_token = [w.translate(table) for w in correct_token]
    correct_token = "".join(correct_token)
    correct_token = correct_token.strip()

    data = {'sentence': sentence, 'correction': '', 'index': []}
    tokenized_sentence = tokenizer.tokenize(data['sentence'])
    predicted_tokens = mask_predict_token(tokenized_sentence)

    print(correct_token)
    print(predicted_tokens)

    if correct_token in predicted_tokens:
        correct += 1
        print("yes")
    else:
        print("now")

    print()

    count += 1
    #pbar.update(1)

    i += 1
    if (i == N):
        break
#pbar.close()

print(correct)
print(count)