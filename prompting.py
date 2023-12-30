import os

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pathlib import Path

from utils.utils import read_data

load_dotenv()

if __name__ == "__main__":
  # path_to_bitexts = Path("data/confiant/bitexts")
  path_to_bitexts = Path("data/tatoeba/bitexts")
  path_to_prompts = Path("prompts")
  path_to_result = Path("results/tatoeba")
  test_examples, = read_data([path_to_bitexts / "eval.fra"])
  model_name = "ft:gpt-3.5-turbo-1106:personal::8O8LT6Ht"
  # model_name = "gpt-4-1106-preview"
  # model_name = "ft:gpt-3.5-turbo-0613:personal::8bY5JdGO"
  exclude_prompts = ["prompt_en", "prompt_fr"]
  batch_size = 1
  total_size = len(test_examples)
  num_batches = total_size // batch_size + int(total_size % batch_size != 0)
  prompts = []
  for promptfile in path_to_prompts.glob("*.txt"):
     with open(promptfile, "r", encoding="utf-8") as fp:
        prompts.append((promptfile.stem, fp.read()))
  
  client = OpenAI(api_key=os.getenv("OPENAI_KEY"), organization=os.getenv("ORGANIZATION_ID"))
  model_path_name = "_".join(model_name.split(":")[1:]) if ':' in model_name else model_name
  path_out = path_to_result / (model_path_name)
  path_out.mkdir(parents=True, exist_ok=True)
  for name, prompt in prompts:
    if name in exclude_prompts:
      continue
    answers = []
    for i in range(num_batches):
    # There is a limit of 4096 tokens per request, so
    # process by batch
      batch = test_examples[(i*batch_size):((i+1)*batch_size)]
      completion = client.chat.completions.create(
        model=model_name,
        seed=0,
        messages=[
          {"role": "system", "content": prompt},
          {"role": "user", "content": "\n".join(batch)}
        ]
      )
      res = completion.choices[0].message.content
      # res = completion.choices[0].message.content.split("\n")
      # assert len(batch) == len(res)
      answers.append(res)
      # answers.extend(res)
    assert len(answers) == len(test_examples)
    df = pd.DataFrame({"request":test_examples, "answer": answers})
    df["answer"] = df["answer"].str.replace("\n", "\\n")
    df["request"] = df["request"].str.replace("\n", "\\n")
    df.to_csv(path_out / f"answer_to_{name}.csv", index=False)