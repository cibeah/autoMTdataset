import argparse

from tokenizers import (
    decoders,
    models, 
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)

parser = argparse.ArgumentParser(description='tokenizer')
parser.add_argument('mode', type=str, help='train or encode or decode')
parser.add_argument('--seq', "-s", type=str, help='sequence to encode or decode')
parser.add_argument('--model-name', "-m", type=str, help='model name', default="tokhf25k_kreyol")

paths_to_input_file = ["data/tokenizer_input.txt"]

if __name__ == "__main__":
  args = parser.parse_args()
  if args.mode == "train":
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=25_000, special_tokens=["<|endoftext|>"])
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(paths_to_input_file, trainer=trainer)
    tokenizer.save("models/tokenizers/" + args.model_name + ".json")
  elif args.mode == "encode":
    tokenizer = Tokenizer.from_file("models/tokenizers/" + args.model_name + ".json")
    encoding = tokenizer.encode(args.seq)
    print("|".join([args.seq[start:stop] for start, stop in encoding.offsets]))
    print(len(encoding.tokens), " TOKENS")
    # print(" | ".join([t.replace("Ġ", "_") for t in encoding.tokens]))
  elif args.mode == "decode":
    tokenizer = Tokenizer.from_file("models/tokenizers/" + args.model_name + ".json")
    decoding = tokenizer.decode(args.seq)
    print(decoding)


# wrapped_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     bos_token="<|endoftext|>",
#     eos_token="<|endoftext|>",
# )