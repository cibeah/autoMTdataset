from pathlib import Path
import shutil

import pandas as pd

from utils import read_data

path_to_synthetic = "results/training-parallel-commoncrawl/data/gemini-1.5-pro-002"
path_to_out = "data/synthetic-commoncrawl/bitexts"

# sorting folders
path_to_done = Path(path_to_synthetic) / "processed"
path_to_pending = Path(path_to_synthetic) / "pending"


count_synth = 0
count_unaligned = 0
remaining_docs = 0
path_to_out = Path(path_to_out)
path_to_out.mkdir(parents=True, exist_ok=True)
path_to_done.mkdir(parents=True, exist_ok=True)
path_to_pending.mkdir(parents=True, exist_ok=True)

def write_bitexts(ref, synth):
    with open(path_to_out / "data.fra", "a", encoding="utf-8") as fp:
        fp.write("\n".join(ref)+"\n")
    with open(path_to_out / "data.mart", "a", encoding="utf-8") as fp:
        fp.write("\n".join(synth)+"\n")


# for path in Path(path_to_synthetic).glob("*.mart"):
#     synth, = read_data([path])
#     aligned_synth = []
#     for line in synth:
#         if line == '""':
#             aligned_synth.extend(['' for _ in range(32)])
#         else:
#             aligned_synth.append(line)
#     with open(path, "w", encoding="utf-8") as fp:
#         fp.write("\n".join(aligned_synth))

# for path in Path(path_to_synthetic).glob("*.csv"):
#     df = pd.read_csv(path)
#     count_synth += df.shape[0]
#     ref, synth = df.iloc[:,0].values, df.iloc[:,1].values
#     write_bitexts(ref, synth)    
#     shutil.move(path, path_to_done / path.name)

for path in Path(path_to_synthetic).glob("*.mart"):
    bitexts = ([], [])
    pending = []
    synth, ref = read_data([path, path.parent / f"{path.stem}.fra"])

    if len(synth) == len(ref):
        # filter out empty strings
        for i, (line, refline) in enumerate(zip(synth, ref)):
            if line == "" and refline == "":
                continue
            if line == "" or line == '""':
                pending.append(refline)
            else:
                bitexts[0].append(refline)
                bitexts[1].append(line)
        ref, synth = bitexts
        count_synth += len(synth)
        write_bitexts(ref, synth)
        if len(pending) > 0:
            with open(path_to_pending / f"{path.stem}.fra", "w", encoding="utf-8") as fp:
                fp.write("\n".join(pending))
        shutil.move(path, path_to_done / path.name)
        shutil.move(path.parent / f"{path.stem}.fra", path_to_done / f"{path.stem}.fra")
    else:
        count_unaligned += len(synth)
        remaining_docs += 1
        print(f"UNALIGNED bitexts for file {path}. Not processing")

print(f"Processed {count_synth} lines.")
print(f"{count_unaligned} unaligned synthetic lines on {remaining_docs} documents")