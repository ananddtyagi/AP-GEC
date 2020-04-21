#written by Anand Tyagi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #removes tf debugging

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def correct(sentences):
    #USE THE m2-correct that is taken from https://github.com/samueljamesbell/m2-correct
    os.system('cd ./m2-correct')

    return

def sim_score(sentence, corrected):
    return np.inner(sentence, corrected)

def sentences_to_embeddings(sentences):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embeddings = embed(sentences)

    return embeddings

def phase2(sentences):
    corrected = correct(sentences)
    sentence_embeddings = sentences_to_embeddings(sentences)
    corrected_embeddings = sentences_to_embeddings(corrected)
    keep = []

    for i in range(len(sentence_embeddings)):
        if sim_score(sentence_embeddings[i], corrected_embeddings[i]) >= 0.87:
            keep.append(i)

    return keep
