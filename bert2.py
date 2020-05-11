from transformers import pipeline
import torch
from transformers import BertModel, BertTokenizer
import string
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

def predict(sentence, threshold):
    tokenized_sentence = tokenizer.tokenize(sentence)
    best_predictions = []
    for i, word in enumerate(tokenized_sentence):
        if tokenized_sentence[i] in string.punctuation: #avoid calculating punctuation which will likely have the highest result
            continue;

        masked_sentence = ' '.join(mask(tokenized_sentence, i)).replace(' ##', '')
        prediction = fill_mask(masked_sentence)[0]
        # print(masked_sentence)
        # print(prediction)
        predicted_token = tokenizer.convert_ids_to_tokens(prediction['token'])
        if predicted_token != word:
            if prediction['score'] > threshold:
                best_predictions.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})

    sorted_best_predictions = sorted(best_predictions, key=itemgetter('confidence'), reverse=True)


    return sorted_best_predictions
