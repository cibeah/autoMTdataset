import pandas as pd
from pathlib import Path
from sacrebleu.metrics import CHRF

from utils import read_data

def score(predictions, references):
    # we use chrf to avoid tokenize choice
    chrf = CHRF()
    return chrf.corpus_score(predictions, references)

def evaluate(ref_path, predictions):
    inputs, references = read_data(ref_path)
    references = [[ref,] for ref in references]
    return score(predictions, references)

if __name__ == "__main__":
    path_to_result = Path("results")
    path_to_prompts = Path("prompts")
    for promptfile in path_to_prompts.glob("*.txt"):
        resultfile = path_to_result / f"answer_to_{promptfile.stem}.csv"
        df = pd.read_csv(resultfile)
        eval = evaluate("data/bitexts", list(df.iloc[:, 0]))
        print(promptfile.stem, ":", eval)