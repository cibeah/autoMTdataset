from pathlib import Path
import shutil

import pandas as pd
import numpy as np

from utils import read_data

path_to_bitexts = Path("data/synthetic-commoncrawl/bitexts")

# sorting folders
path_to_lang1 = Path(path_to_bitexts) / "data.fra"
path_to_lang2 = Path(path_to_bitexts) / "data.mart"

# 1. filter empty lines
# 2. filter lines with same sentence right and left
# 3. filter lines too long for either fra or mart

text_lang1, text_lang2 = read_data([path_to_bitexts / "data.fra", path_to_bitexts / "data.mart"])
text_lang1_clean, text_lang2_clean = [], []

empty = same = too_long = 0
len1 = []
len2 = []

for line_lang1, line_lang2 in zip(text_lang1, text_lang2):
    # 1. filter empty lines
    if len(line_lang1) == 0 or len(line_lang2) == 0:
        empty += 1
        continue
    # 2. filter lines with same sentence right and left
    if line_lang1 == line_lang2:
        same += 1
        continue
    
    text_lang1_clean.append(line_lang1)
    len1.append(len(line_lang1))
    text_lang2_clean.append(line_lang2)
    len2.append(len(line_lang2))

mean_lang1, std_lang1 = np.mean(len1), np.std(len1)
mean_lang2, std_lang2 = np.mean(len2), np.std(len2)

# 3. filter lines too long for either fra or mart
threshold1 = mean_lang1 + 5 * std_lang1
threshold2 = mean_lang2 + 5 * std_lang2

text_lang1, text_lang2 = text_lang1_clean, text_lang2_clean
text_lang1_clean, text_lang2_clean = [], []
for line_lang1, line_lang2 in zip(text_lang1, text_lang2):
    if len(line_lang1) >= threshold1 or len(line_lang2) >= threshold2:
        too_long += 1
        continue
    text_lang1_clean.append(line_lang1)
    text_lang2_clean.append(line_lang2)

# write clean data
with open(path_to_bitexts / "data_clean.fra", "a", encoding="utf-8") as fp:
    fp.write("\n".join(text_lang1_clean)+"\n")
with open(path_to_bitexts / "data_clean.mart", "a", encoding="utf-8") as fp:
    fp.write("\n".join(text_lang2_clean)+"\n")

print(f"Removed {empty + same + too_long} sentences: {empty} empty, {same} identical pairs, {too_long} too long.")
print(f"Wrote clean bitexts at {path_to_bitexts}.")