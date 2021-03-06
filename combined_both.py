from transformers import pipeline
import torch
from transformers import BertModel, BertTokenizer, ElectraForPreTraining
import string
from operator import itemgetter
from nltk.corpus import wordnet

topk = 5

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
discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")

def are_syns(word1, word2):
    for synset in wordnet.synsets(word1):
        for lemma in synset.lemma_names():
            if lemma == word2 and lemma != word1:
                return True
    return False

def discriminator_predict(input_sent):
    fake_sentence = tokenizer.tokenize(input_sent)
    fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
    fake_tokens = tokenizer.convert_ids_to_tokens(fake_inputs[0])
    discriminator_outputs = discriminator(fake_inputs)
    sorted_outputs_indices = torch.argsort(discriminator_outputs[0])
    top_k_tokens = []
    top_k_indices = []
    for i in range(topk):
        if i == len(fake_tokens):
            break;
        top_k_tokens.append(fake_tokens[sorted_outputs_indices[-i-1]])
        top_k_indices.append(sorted_outputs_indices[-i-1]-1)
    return(top_k_indices)

def predict(sentence, threshold):
    tokenized_sentence = tokenizer.tokenize(sentence)
    combined = []
    # combined2 = []

    disc_idx = discriminator_predict(sentence)

    for i in disc_idx:
        masked_sentence = ' '.join(mask(tokenized_sentence, i)).replace(' ##', '')

        prediction = fill_mask(masked_sentence)[0]
        predicted_token = tokenizer.convert_ids_to_tokens(prediction['token'])
        try:
            if predicted_token != tokenized_sentence[i] and not are_syns(predicted_token, tokenized_sentence[i]):
                combined.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})
            # if prediction['score'] > threshold:
            #     combined2.append({'index': i, 'token': predicted_token, 'confidence': prediction['score']})
        except IndexError:
            print('index error')
            print(tokenized_sentence)
            print(i)
            return []
    sorted_combined = sorted(combined, key=itemgetter('confidence'), reverse=True)
    # sorted_combined2 = sorted(combined2, key=itemgetter('confidence'), reverse=True)
    return [sug['token'] for sug in sorted_combined[:topk]]
    # return [sug['token'] for sug in sorted_combined[:topk]], [sug['token'] for sug in sorted_combined2]