import pandas as pd
from pathlib import Path
from sacrebleu.metrics import CHRF, BLEU
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from tokenizers import Tokenizer

from utils import read_data
# from utils.utils import read_data

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

def evaluate(ref_path, predictions):
    references, = read_data(ref_path)
    references = [references]
    return score(predictions, references)

if __name__ == "__main__":
    dataset = "confiant", "mart"
    path_to_result = Path(f"results/{dataset[0]}")
    path_to_prompts = Path("prompts")
    path_to_eval = Path(f"data/{dataset[0]}/bitexts/eval.{dataset[1]}")
    for path_model in path_to_result.glob("*"):
        print(f"******MODEL: {path_model.name}******")
        for promptfile in path_to_prompts.glob("*.txt"):
            resultfile = path_model / f"answer_to_{promptfile.stem}.csv"
            try:
                df = pd.read_csv(resultfile)
            except:
                continue
            eval = evaluate([path_to_eval], list(df["answer"]))
            print(promptfile.stem, ":", eval)