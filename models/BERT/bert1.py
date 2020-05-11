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

top_k = 5

def predict(tokenized_sentence):
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
    for k in range(min(top_k, len(orignal_sentence))):
        highest_confidence_corrections.append(sorted_best_predictions[k])

    top_k_tokens = []
    for ele in highest_confidence_corrections:
        top_k_tokens.append(tokenizer.convert_ids_to_tokens(ele['token']))

    # for ele in sorted_best_predictions:
    #     print(ele)

    #toke sentence with highest confidence correction without special tokens
    return(top_k_tokens)

