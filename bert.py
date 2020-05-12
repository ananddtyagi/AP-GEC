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

def predict(sentence, topk, threshold):
    tokenized_sentence = tokenizer.tokenize(sentence)
    bert1 = []
    bert2 = []
    for i, word in enumerate(tokenized_sentence):
        if tokenized_sentence[i] in string.punctuation: #avoid calculating punctuation which will likely have the highest result
            continue;

        masked_sentence = ' '.join(mask(tokenized_sentence, i)).replace(' ##', '')
        prediction = fill_mask(masked_sentence)[0]
        # print(masked_sentence)
        # print(prediction)
        predicted_token = tokenizer.convert_ids_to_tokens(prediction['token'])
        if predicted_token != word:
            bert1.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})
            if prediction['score'] > threshold:
                bert2.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})
    sorted_bert1 = sorted(bert1, key=itemgetter('confidence'), reverse=True)
    sorted_bert2 = sorted(bert2, key=itemgetter('confidence'), reverse=True)

    return [sug['token'] for sug in sorted_bert1[:topk]], [sug['token'] for sug in sorted_bert2]
