import pandas as pd
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from utils.utils import read_data

# First tokenize the input and outputs.
# The format below is how the model was trained so the input should be "Sentence </s> SRCCODE". 
# Similarly, the output should be "TGTCODE Sentence </s>". 
# Example: For Saint Lucian Patois to English translation, we need to use language indicator tags: 
# <2acf> and <2eng> where acf represents Saint Lucian Patois and eng represents English.
# For a mapping of the original language and language code (3 character) 
# to mBART-50 compatible language tokens consider the dictmap dictionary:
# Note: We mapped languages to their language tokens manually. 
# For example, we used en_XX, fr_XX, es_XX for English, French and Spanish as in the original mBART-50 model. 
# But then we repurposed other tokens for Creoles.


dictmap_from_mbart = {'acf': 'ar_AR', 'eng': 'en_XX', 'fra': 'fr_XX', 'gcf': 'ja_XX', 'gcr': 'sv_SE', 'hat': 'ko_KR', 'kea': 'my_MM', 'mart': 'ro_RO', 'rcf': 'km_KH'}
dictmap_from_scratch = {'acf': '<2acf>', 'eng': '<2eng>', 'fra': '<2fra>', 'gcf': '<2gcf>', 'gcr': '<2gcr>', 'hat': '<2hat>', 'kea': '<2kea>', 'mart': '<2mart1259>', 'rcf': '<2rcf>'}

if __name__ == "__main__":
    model_name = "jhu-clsp/kreyol-mt-scratch-pubtrain" #"jhu-clsp/kreyol-mt-pubtrain"
    # model_name = "jhu-clsp/kreyol-mt"
    dataset, split = "restog", "data"
    # dataset, split = "kreyolmt-public-mart", "test"
    # dataset, split = "kreyolmt-public-gcr", "test"
    lang2lang = "fra", "mart"
    dictmap = dictmap_from_scratch if model_name.find("scratch") > -1 else dictmap_from_mbart
    path_to_bitexts = Path(f"data/{dataset}/bitexts")
    path_to_result = Path(f"results/{dataset}/{split}")
    path_out = path_to_result / (model_name.replace("/", "_"))
    path_out.mkdir(parents=True, exist_ok=True)
    
    test_examples, = read_data([path_to_bitexts / f"{split}.fra"])
    n_data = len(test_examples)
    batch_size = 32
    num_batches = n_data // batch_size + int(n_data % batch_size != 0)

    # get model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval() # set dropouts to zero

    predictions = []
    # for i in range(n_data):
    for i in range(num_batches):
        batch = test_examples[(i*batch_size):((i+1)*batch_size)]
        input_seq = [f"{sample} </s> {dictmap[lang2lang[0]]}" for sample in batch]
        inp = tokenizer(input_seq, add_special_tokens=False, return_tensors="pt", padding=True)
        input_ids = inp.input_ids
        model_output = model.generate(
            input_ids,
            use_cache=True,
            num_beams=4,
            max_length=60,
            min_length=1,
            early_stopping=True,
            pad_token_id=tokenizer.added_tokens_encoder["<pad>"],
            bos_token_id=tokenizer.added_tokens_encoder["<s>"],
            eos_token_id=tokenizer.added_tokens_encoder["</s>"],
            decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc(dictmap[lang2lang[1]])
        )
        decoded_output = tokenizer.batch_decode(
            model_output, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        predictions.extend(decoded_output)

    df = pd.DataFrame({"request": test_examples, "answer": predictions})
    df["answer"] = df["answer"].str.replace("\n", "\\n")
    df["request"] = df["request"].str.replace("\n", "\\n")
    df.to_csv(path_out / f"predictions.csv", index=False)