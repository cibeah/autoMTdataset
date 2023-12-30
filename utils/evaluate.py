import pandas as pd
from pathlib import Path
from sacrebleu.metrics import CHRF

from utils import read_data
# from utils.utils import read_data

def score(predictions, references):
    # we use chrf to avoid tokenize choice
    chrf = CHRF()
    return chrf.corpus_score(predictions, references)

def evaluate(ref_path, predictions):
    references, = read_data(ref_path)
    references = [[ref,] for ref in references]
    return score(predictions, references)

if __name__ == "__main__":
    path_to_result = Path("results/tatoeba")
    path_to_prompts = Path("prompts")
    path_to_eval = Path("data/tatoeba/bitexts/eval.gcf")
    for path_model in path_to_result.glob("*"):
        print(f"******MODEL: {path_model.stem}******")
        for promptfile in path_to_prompts.glob("*.txt"):
            resultfile = path_model / f"answer_to_{promptfile.stem}.csv"
            try:
                df = pd.read_csv(resultfile)
            except:
                continue
            eval = evaluate([path_to_eval], list(df["answer"]))
            print(promptfile.stem, ":", eval)         