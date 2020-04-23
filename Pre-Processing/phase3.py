#This code is from m2-correct that is taken from https://github.com/samueljamesbell/m2-correct and was modified by Pranav

import argparse

import parser


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to text annotated with corrections')
    parser.add_argument('outputpath', help='Path to corrected output text')
    return parser.parse_args()


def apply_corrections(sentence, corrections):
    """Return a new sentence with corrections applied.

    Sentence should be a whitespace-separated tokenised string. Corrections
    should be a list of corrections.
    """
    tokens = sentence.split(' ')
    offset = 0

    all_sentences = []

    for c in corrections:
        tokens, offset = _apply_correction(tokens, c, offset)
        all_sentences.append(' '.join(tokens))

    return all_sentences


def _apply_correction(tokens, correction, offset):
    """Apply a single correction to a list of tokens."""
    start_token_offset, end_token_offset, _, insertion = correction
    to_insert = insertion[0].split(' ')
    end_token_offset += (len(to_insert) - 1)

    to_insert_filtered = [t for t in to_insert if t != '']

    head = tokens[:start_token_offset + offset]
    tail = tokens[end_token_offset + offset:]

    new_tokens = head + to_insert_filtered + tail

    new_offset = len(to_insert_filtered) - (end_token_offset - start_token_offset) + offset

    return new_tokens, new_offset


def phase3():

    with open('./A.train.gold.bea19.m2', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    sentences, corrections = parser.parse(lines)

    with open('./output.txt', 'w', encoding='utf-8') as f:
        for s, c in zip(sentences, corrections):
            corrected = apply_corrections(s, c[0])
            for correct in corrected:
                f.write(correct + "\n")

phase3()