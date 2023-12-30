import numpy as np
from pathlib import Path


def write_helper(path, content):
  with open(path, "w", encoding="utf-8") as fp:
    fp.write(content)

def read_data(path_to_bitexts):
  data = []
  for path in path_to_bitexts:
    with open(path, "r", encoding="utf-8") as fp:
        data.append(fp.read().split("\n"))
  return data

def print_translations(inputs, translations):
  print("\n".join([inputs[i] + " => " + translations[i] for i in range(len(inputs))]))

def train_test_split(path_to_bitexts, train=0.8, eval=0.1, test=0.1):
  data = read_data(path_to_bitexts)
  size = len(data[0])
  for chunk in data:
    assert size == len(chunk), "All languages must have the same number of sentences."
  idx = np.random.choice(np.arange(size), size, replace=False)
  num_train, num_eval = int(train*size), int(eval*size)

  train_idx = idx[:num_train]
  eval_idx, test_idx = idx[num_train:(num_train+num_eval)], idx[num_train+num_eval:]

  for path, chunk in zip(path_to_bitexts, data):
    for name, pts in zip(["train", "eval", "test"], [train_idx, eval_idx, test_idx]):
      output = [chunk[i] for i in pts]
      # Save
      if len(output) > 0:
        write_helper(Path(path).parent / f"{name}{Path(path).suffix}", "\n".join(output))