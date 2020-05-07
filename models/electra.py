from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="google/electra-base-generator",
    tokenizer="google/electra-base-generator"
)

print(
    fill_mask(f"HuggingFace is creating a [MASK] that the community uses to solve NLP tasks.")
)


print('+++++++++++++++++++')


print('+++++++++++++++++++')
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

sentence = "The quick brown fox jumps over the lazy dog"
fake_sentence = "The quick brown fox fake over the lazy dog"

fake_tokens = tokenizer.tokenize(fake_sentence)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
discriminator_outputs = discriminator(fake_inputs)
predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

[print("%7s" % token, end="") for token in fake_tokens]
print('\n')
[print("%7s" % int(prediction), end="") for prediction in predictions.tolist()]
print('\n')
