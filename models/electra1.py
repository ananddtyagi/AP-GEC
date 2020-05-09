from transformers import pipeline
import torch
from transformers import ElectraModel, ElectraTokenizer
import string

def fstr(sentence):
    return eval(f"f'{sentence}'")

def mask(sentence, masked_index):
    return sentence[:masked_index] + ['[MASK]'] + sentence[masked_index+1:]


fill_mask = pipeline(
    "fill-mask",
    model="google/electra-base-generator",
    tokenizer="google/electra-base-generator"
)

# print(
#     fill_mask(f"HuggingFace is creating a [MASK] that the community uses to solve NLP tasks.")
# )
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-generator')


data = {'sentence': 'HuggingFace is creating an tool that the community uses to solve NLP tasks.', 'correction': '', 'index': []}
tokenized_sentence = tokenizer.tokenize(data['sentence'])
max = {'index': 0, 'token': 0, 'confidence': 0} #index, token, confidence

for i, word in enumerate(tokenized_sentence):

    if tokenized_sentence[i] in string.punctuation: #avoid calculating punctuation which will likely have the highest result
        continue;

    masked_sentence = ' '.join(mask(tokenized_sentence, i)).replace(' ##', '')
    prediction = fill_mask(masked_sentence)[0]
    print(prediction)
    print(word)
    if tokenizer.convert_ids_to_tokens(prediction['token']) != word:
        if prediction['score'] > max['confidence']:
            max = {'index': i, 'token': prediction['token'], 'confidence': prediction['score']}


    print(tokenizer.convert_ids_to_tokens(max['token']))
