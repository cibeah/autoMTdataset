import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from sacrebleu.metrics import CHRF, BLEU
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from tokenizers import Tokenizer

from utils import read_data
# from utils.utils import read_data

parser = argparse.ArgumentParser(description='tokenizer')
parser.add_argument('--cross', action="store_true")

class TokenizerKreyol:
    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file(path)
    def __call__(self, seq):
        return " ".join([seq[start:stop] for start, stop in self.tokenizer.encode(seq).offsets])

BASE_TOKENIZER = Tokenizer13a()
KREYOL_TOKENIZER = TokenizerKreyol("models/tokenizers/tokhf25k_kreyol.json")

def score(predictions, references, tokenizer=KREYOL_TOKENIZER):
    # chrf++ avoids tokenize choice
    chrf = CHRF(word_order=0).corpus_score(predictions, references)
    chrfpp = CHRF(word_order=2).corpus_score(predictions, references)
    bleu = BLEU().corpus_score(predictions, references)
    # print(chrf, chrfpp, bleu)
    custom_tokenized = (
        [tokenizer(pred) for pred in predictions],
        [[tokenizer(ref) for ref in list_ref] for list_ref in references]
    )
    bleu_custom = BLEU(tokenize="none", force=True).corpus_score(*custom_tokenized)
    return {
        "chrF": round(chrf.score,2),
        "chrF++": round(chrfpp.score,2), 
        "BLEU": round(bleu.score,2),
        "BLEU_KreyTok": round(bleu_custom.score,2),
    }

def avg_scores(score_list):
    score_agg = {}
    for score_dict in score_list:
        for score_name in score_dict:
            value = score_agg.get(score_name, [])
            value.append(score_dict[score_name])
            score_agg[score_name] = value
    return {
        name: (round(float(np.mean(value)),2), round(float(np.std(value)),2)) 
        for name, value in score_agg.items()
    }

def evaluate(ref_path, predictions, cross_eval=False, num_batches=5):
    references, = read_data(ref_path)
    references = [references]
    if cross_eval:
        num_predictions = len(predictions)
        batch_size = num_predictions // num_batches
        indices = np.random.choice(np.arange(num_predictions), size=num_predictions, replace=False)
        scores = []
        for i in range(num_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            batch_predictions = [predictions[ind] for ind in batch_indices]
            batch_refs = [[refs[ind] for ind in batch_indices] for refs in references]
            if batch_predictions:
                scores.append(score(batch_predictions, batch_refs))
        return avg_scores(scores)
    return score(predictions, references)


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(0)
    dataset = "confiant", "mart"
    split = "test"
    path_to_result = Path(f"results/{dataset[0]}/{split}")
    path_to_prompts = Path("prompts")
    path_to_eval = Path(f"data/{dataset[0]}/bitexts/{split}.{dataset[1]}")
    scores = []
    for path_model in path_to_result.glob("*"):
        print(f"******MODEL: {path_model.name}******")
        for promptfile in path_to_prompts.glob("*.txt"):
            resultfile = path_model / f"answer_to_{promptfile.stem}.csv"
            try:
                df = pd.read_csv(resultfile)
            except:
                continue
            eval = evaluate([path_to_eval], list(df["answer"]), cross_eval=args.cross)
            print(promptfile.stem, ":", eval)
            eval.update({"model": path_model.stem, "prompt": promptfile.stem})
            scores.append(eval)
    table = pd.DataFrame(scores)
    table.to_csv(path_to_result / "scores.csv", index=False)