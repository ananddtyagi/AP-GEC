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

def mask_predict_sent(tokenized_sentence):
    best_predictions = []
    orignal_sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_sentence))

    min_confidence = 0.80

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
            if prediction['score'] > min_confidence:
                prediction['no_special_token_sentence'] = no_special_tokens_sentence
                prediction['index'] = i
                best_predictions.append(prediction)

                print(word + "/" + tokenizer.convert_ids_to_tokens(prediction['token']))

    for prediction in best_predictions:
        tokenized_sentence[prediction['index']] = tokenizer.convert_ids_to_tokens(prediction['token'])

    sentence = ' '.join(tokenized_sentence).replace(' ##', '')

    # for ele in sorted_best_predictions:
    #     print(ele)

    #toke sentence with highest confidence correction without special tokens
    return(sentence)

#works badly with punctuation and spacing.
#maybe an index issue? Like it's replacing the predicted token at the wrong index?
#hmmm how should we fix it?

sentence = "I have my own plan too but I do n't same to them , I want to become a Journalist ."
data = {'sentence': sentence, 'correction': '', 'index': []}
tokenized_sentence = tokenizer.tokenize(data['sentence'])

new_sent = mask_predict_sent(tokenized_sentence)
print(new_sent)
