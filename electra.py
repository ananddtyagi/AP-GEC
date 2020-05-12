from transformers import pipeline
import torch
from transformers import ElectraModel, ElectraTokenizer, ElectraForPreTraining
import string
from operator import itemgetter

def fstr(sentence):
    return eval(f"f'{sentence}'")

def mask(sentence, masked_index):
    return sentence[:masked_index] + ['[MASK]'] + sentence[masked_index+1:]


fill_mask = pipeline(
    "fill-mask",
    model="google/electra-large-generator",
    tokenizer="google/electra-large-generator"
)


tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-generator')
discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")


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
            # discriminator_outputs = discriminator(fake_inputs)
            # predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

    sorted_electra1 = sorted(electra1, key=itemgetter('confidence'), reverse=True)
    sorted_electra2 = sorted(electra2, key=itemgetter('confidence'), reverse=True)

    return [sug['token'] for sug in sorted_electra1[:topk]], [sug['token'] for sug in sorted_electra2]

