import argparse
from pathlib import Path
import re

import numpy as np

EXCLUDE_PATTERNS = [
    "the ",
    " is ",
    " and ",
    " can ",
    " to ",
    " of ",
    " und ",
    " wir ",
    " wie ",
    " ein ",
    " eine ",
    " one ",
    " an ",
    " das ",
    " die ",
    " you ",
    " that "
]
ALPHANUMERIC_PATTERN = r'^[a-zA-Z]+$'

parser = argparse.ArgumentParser(description='tokenizer')
parser.add_argument('path', type=str, help="path to data to clean")

def clean_data(path_to_tgt_data, subsample=0.3):
    """
    Following the Kréyol-MT cleaning guidelines
    + some specific pattern matching to remove non-french sentences
    """
    with open(path_to_tgt_data, "r", encoding="utf-8") as fp:
        data = fp.read().split("\n")
    # remove sentences with "is "
    # remove sene
    path_to_tgt_data = Path(path_to_tgt_data)
    data_size = len(data)
    num_samples = int(np.ceil(subsample*data_size))
    path_to_cleaned_data = path_to_tgt_data.parent / f"cleaned_{path_to_tgt_data.stem}"
    path_to_removed_data = path_to_tgt_data.parent / f"removed_{path_to_tgt_data.stem}"
    path_to_sampled_data = path_to_tgt_data.parent / f"sampled_{path_to_tgt_data.stem}"
    cleaned_ = set()
    lengths = []
    removed = []
    print(f"Reading {data_size} sentences from {path_to_tgt_data}")

    ## clean 
    for sentence in data:
        sentence = sentence.strip(" ")
        words = sentence.split(" ")

        # remove with less than 3 words (and empty sentences)
        if len(words) <= 3:
            removed.append(sentence)
            continue
        
        # remove sentences with too many non-alphabetic characters
        if np.mean([
            re.match(ALPHANUMERIC_PATTERN, word) is not None
            for word in words
        ]) < 0.5:
            removed.append(sentence)
            continue

        # remove sentences with patterns
        if any([pattern in sentence.lower() for pattern in EXCLUDE_PATTERNS]):
            removed.append(sentence)
            continue

        cleaned_.add((sentence, len(sentence)))
        lengths.append(len(sentence))
    
    lengths = np.array(lengths)
    avg_length, std_length = np.mean(lengths), np.std(lengths)
    sentence_too_long = lengths > ( avg_length + 5 * std_length)
    
    print(
        f"{len(cleaned_)} sentences after pattern matching cleaning. \n"
        f"Average length: {avg_length} - Standard Deviation: {std_length} \n"
        f"{np.sum(sentence_too_long)} sentences too long (> avg + 5 std(lengths))"
    )

    # remove too long sentences
    cleaned = []
    for i, (sentence, length) in enumerate(cleaned_):
        if length > avg_length + 5 * std_length:
            removed.append(sentence)
        else:
            cleaned.append(sentence)
    cleaned_size = len(cleaned)
    print(f"{cleaned_size} sentences after cleaning")
    assert cleaned_size > num_samples, "Not enough remaining sentences after cleaning"

    with open(path_to_cleaned_data, "a", encoding="utf-8") as fp:
        fp.write("\n".join(cleaned))
    
    ## subsample
    indices = np.random.choice(cleaned_size, num_samples, replace=False)
    subsampled = [cleaned[i] for i in indices]

    with open(path_to_sampled_data, "a", encoding="utf-8") as fp:
        fp.write("\n".join(subsampled))
    
    with open(path_to_removed_data, "a", encoding="utf-8") as fp:
        fp.write("\n".join(removed))

if __name__ == "__main__":
    # path_to_target_lang_data = "data/training-parallel-commoncrawl/commoncrawl.fr-en.fr"
    args = parser.parse_args()
    path_to_target_lang_data = args.path
    clean_data(path_to_target_lang_data)