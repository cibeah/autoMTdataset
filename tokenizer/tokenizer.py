import argparse

import sentencepiece as spm
import os


parser = argparse.ArgumentParser(description='tokenizer')
parser.add_argument('mode', type=str, help='train or encode or decode')
parser.add_argument('--seq', "-s", type=str, help='sequence to encode or decode')
parser.add_argument('--model-name', "-m", type=str, help='model name', default="tok25k_kreyol")

path_to_input_file = "C:\\Users\\clair\\Documents\\Projects\\creole-nlp\\nlp-kreyol-resources\\datasets\\kreyolad\\monolingual.mart"

# from Karpathy's colab 
# https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing#scrollTo=rSv1vfIVOhvr
options = dict(
  # input spec
  input=path_to_input_file,
  input_format="text",
  # output spec
  model_prefix="tok25k_kreyol", # output filename prefix
  # algorithm spec
  # BPE alg
  model_type="bpe",
  vocab_size=8_000,
  # normalization
  normalization_rule_name="identity",
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences
  max_sentence_length=4192, # max number of bytes per sentence
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.99995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)

def load(name):
  sp = spm.SentencePieceProcessor()
  sp.load("models/tokenizers/" + name + ".model")
  return sp

if __name__ == "__main__":
  args = parser.parse_args()
  if args.mode == "train":
    options["model_prefix"] = "models/tokenizers/" + args.model_name
    spm.SentencePieceTrainer.train(**options)
  elif args.mode == "encode":
    sp = load(args.model_name)
    tokens = sp.encode(args.seq)
    print(tokens)
    print([(sp.id_to_piece(idx)) for idx in tokens])
  elif args.mode == "decode":
    sp = load(args.model_name)
    seq = args.seq.split(" ")
    print("".join([sp.id_to_piece(int(idx)) for idx in seq]))