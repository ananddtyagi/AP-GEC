from transformers import pipeline
import torch
from transformers import ElectraModel, ElectraTokenizer, ElectraForPreTraining
import string
from operator import itemgetter
from nltk.corpus import wordnet


def fstr(sentence):
    return eval(f"f'{sentence}'")

def mask(sentence, masked_index):
    return sentence[:masked_index] + ['[MASK]'] + sentence[masked_index+1:]



fill_mask = pipeline(
    "fill-mask",
    model="google/electra-large-generator",
    tokenizer="google/electra-large-generator"
)

tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-generator')


def are_syns(word1, word2):
    for synset in wordnet.synsets(word1):
        for lemma in synset.lemma_names():
            if lemma == word2 and lemma != word1:
                return True
    return False

def predict(sentence, threshold):
    tokenized_sentence = tokenizer.tokenize(sentence)
    electra = []


    for i, word in enumerate(tokenized_sentence):
        if tokenized_sentence[i] in string.punctuation: #avoid calculating punctuation which will likely have the highest result
            continue;

        masked_sentence = ' '.join(mask(tokenized_sentence, i)).replace(' ##', '')
        prediction = fill_mask(masked_sentence)[0]
        # print(masked_sentence)
        # print(prediction)
        predicted_token = tokenizer.convert_ids_to_tokens(prediction['token'])
        if predicted_token != word and not are_syns(predicted_token, word):
            if prediction['score'] > threshold:
                electra.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})

    sorted_electra = sorted(electra, key=itemgetter('confidence'), reverse=True)

    return [sug['token'] for sug in sorted_electra]