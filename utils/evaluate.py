import argparse
from typing import Iterable

# import evaluate
import numpy as np
import pandas as pd
from pathlib import Path
from sacrebleu.metrics import CHRF, BLEU
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from utils import read_data
# from utils.utils import read_data

parser = argparse.ArgumentParser(description='tokenizer')
parser.add_argument('--cross', action="store_true")

class CustomTokenizer:
    def __init__(self, tokenizer):
        if isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer = tokenizer.sp_model
            self.id_to_token = self.tokenizer.id_to_piece
        else:
            self.tokenizer = Tokenizer.from_file(tokenizer)
            self.id_to_token = self.tokenizer.id_to_token
    
    def __call__(self, seq):
        tokens = self.tokenizer.encode(seq)
        if not isinstance(tokens, Iterable):
            tokens = tokens.ids
        return " ".join([self.id_to_token(idx) for idx in tokens])

BASE_TOKENIZER = Tokenizer13a()
KREYOL_TOKENIZER = CustomTokenizer("models/tokenizers/tokhf25k_kreyol.json")
jhu_auto_tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/kreyol-mt", do_lower_case=False, use_fast=False, keep_accents=True)
JHU_KREYOL_TOKENIZER = CustomTokenizer(jhu_auto_tokenizer)

def score(predictions, references, tokenizers=[KREYOL_TOKENIZER, JHU_KREYOL_TOKENIZER]):
    # chrf++ avoids tokenize choice
    chrf = CHRF(word_order=0).corpus_score(predictions, references)
    chrfpp = CHRF(word_order=2).corpus_score(predictions, references)
    bleu = BLEU().corpus_score(predictions, references)
    # bleu_hf = evaluate.load("bleu").compute(predictions=predictions, references=references)

    # print(chrf, chrfpp, bleu)
    score_dict = {
        "chrF": round(chrf.score,2),
        "chrF++": round(chrfpp.score,2), 
        "BLEU": round(bleu.score,2),
    }
    for i, tokenizer in enumerate(tokenizers):
        custom_tokenized = (
            [tokenizer(pred) for pred in predictions],
            [[tokenizer(ref) for ref in list_ref] for list_ref in references]
        )
        bleu_custom = BLEU(tokenize="none", force=True).corpus_score(*custom_tokenized)
        score_dict.update({f"BLEU_KreyTok{i}": round(bleu_custom.score,2)})
    return score_dict

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
    # dataset, suffix, split = "kreyolmt-public-mart", "mart", "test"
    dataset, suffix, split = "restog", "mart", "data"
    path_to_result = Path(f"results/{dataset}/{split}")
    path_to_eval = Path(f"data/{dataset}/bitexts/{split}.{suffix}")
    scores = []
    for path_model in path_to_result.glob("*"):
        if not path_model.is_dir():
            continue
        print(f"******MODEL: {path_model.name}******")
        for resultfile in path_model.glob("*.csv"):
            try:
                df = pd.read_csv(resultfile)
            except:
                continue
            eval = evaluate([path_to_eval], list(df["answer"]), cross_eval=args.cross)
            print(resultfile.stem, ":", eval)
            eval.update({"model": path_model.stem, "prompt": path_model.stem})
            scores.append(eval)
    table = pd.DataFrame(scores)
    table.to_csv(path_to_result / "scores.csv", index=False)