from pathlib import Path

def read_data(path_to_bitexts):
  path_to_bitexts = Path(path_to_bitexts)
  with open(path_to_bitexts / "data.fra", "r", encoding="utf-8") as fp:
      fr = fp.read().split("\n")
  with open(path_to_bitexts / "data.mart", "r", encoding="utf-8") as fp:
      mq = fp.read().split("\n")
  return fr, mq

def print_translations(inputs, translations):
  print("\n".join([inputs[i] + " => " + translations[i] for i in range(len(inputs))]))