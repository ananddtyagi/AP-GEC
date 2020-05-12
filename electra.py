from transformers import pipeline
import torch
from transformers import ElectraModel, ElectraTokenizer
import string
from operator import itemgetter

def fstr(sentence):
    return eval(f"f'{sentence}'")

def mask(sentence, masked_index):
    return sentence[:masked_index] + ['[MASK]'] + sentence[masked_index+1:]


fill_mask = pipeline(
    "fill-mask",
    model="google/electra-base-generator",
    tokenizer="google/electra-base-generator"
)


tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-generator')

def predict(sentence, topk, threshold):
    tokenized_sentence = tokenizer.tokenize(sentence)
    electra1 = []
    electra2 = []
    for i, word in enumerate(tokenized_sentence):
        if tokenized_sentence[i] in string.punctuation: #avoid calculating punctuation which will likely have the highest result
            continue;

        masked_sentence = ' '.join(mask(tokenized_sentence, i)).replace(' ##', '')
        prediction = fill_mask(masked_sentence)[0]
        # print(masked_sentence)
        # print(prediction)
        predicted_token = tokenizer.convert_ids_to_tokens(prediction['token'])
        if predicted_token != word:
            electra1.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})
            if prediction['score'] > threshold:
                electra2.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})
    sorted_electra1 = sorted(electra1, key=itemgetter('confidence'), reverse=True)
    sorted_electra2 = sorted(electra2, key=itemgetter('confidence'), reverse=True)

    return sorted_electra1[:topk], sorted_electra2
