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

    all_sentence_edit_pairs = []

    for c in corrections:
        prev_tokens = tokens
        tokens, offset, original_edit_sent = _apply_correction(prev_tokens, c, offset)

        all_sentence_edit_pairs.append(["S " + ' '.join(prev_tokens), original_edit_sent])

    return all_sentence_edit_pairs


def _apply_correction(tokens, correction, offset):
    """Apply a single correction to a list of tokens."""
    start_token_offset, end_token_offset, _, insertion, original_edit = correction
    to_insert = insertion[0].split(' ')
    end_token_offset += (len(to_insert) - 1)

    to_insert_filtered = [t for t in to_insert if t != '']

    head = tokens[:start_token_offset + offset]
    tail = tokens[end_token_offset + offset:]

    new_tokens = head + to_insert_filtered + tail

    new_offset = len(to_insert_filtered) - (end_token_offset - start_token_offset) + offset

    return new_tokens, new_offset, original_edit


def _main():
    args = _parse_args()

    with open(args.path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    sentences, corrections = parser.parse(lines)

    with open(args.outputpath, 'w', encoding='utf-8') as f:
        for s, c in zip(sentences, corrections):
            print(c)
            corrected = apply_corrections(s, c[0])
            for correct in corrected:
                f.write(correct[0] + "\n")
                f.write(correct[1] + "\n\n")

if __name__ == '__main__':
    _main()
