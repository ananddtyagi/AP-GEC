from transformers import pipeline
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import string

def fstr(sentence):
    return eval(f"f'{sentence}'")

def mask(sentence, masked_index):
    return sentence[:masked_index] + ['[MASK]'] + sentence[masked_index+1:]

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased",
    tokenizer="bert-base-uncased"
)
#
# print(
#     fill_mask(f"HuggingFace is creating a [MASK] that the community uses to solve NLP tasks.")
# )

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

data = {'sentence': 'HuggingFace is creating an tool that the community uses to solve NLP tasks.', 'correction': '', 'index': []}
tokenized_sentence = tokenizer.tokenize(data['sentence'])
for i, word in enumerate(tokenized_sentence):
    max = {'index': 0, 'token': 0, 'confidence': 0} #index, token, confidence

    if tokenized_sentence[i] in string.punctuation: #avoid calculating punctuation which will likely have the highest result
        continue;

    masked_sentence = ' '.join(mask(tokenized_sentence, i)).replace(' ##', '')
    prediction = fill_mask(masked_sentence)[0]
    if prediction['score'] > max['confidence']:
        max = {'index': i, 'token': prediction['token'], 'confidence': prediction['score']}

        print(prediction)
print(tokenizer.convert_ids_to_tokens(max['token']))