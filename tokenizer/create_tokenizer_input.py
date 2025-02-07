from pathlib import Path
from yaml import load, BaseLoader

if __name__ == "__main__":

    with open("data/tokenizers_input.yaml", "r") as f:
        paths_to_input_file = load(f, BaseLoader)
    for path in paths_to_input_file:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        print(Path(path).parent.parent.stem, "-", Path(path).parent.stem, ": ", len(text))
        with open("data/tokenizer_input.txt", "a", encoding="utf-8") as f:
            f.write(text)