import argparse
import os

from dotenv import load_dotenv
# import google.generativeai as genai
from openai import OpenAI
import pandas as pd
from pathlib import Path
from transformers import pipeline

from utils.gemini_api import GeminiAPI
from utils.utils import read_data

load_dotenv()

PROMPTS_TO_EXCLUDE = [
  "prompt_fr", 
  "prompt_test", 
  "prompt_short", 
  "prompt_medium", 
  "prompt_medium_2", 
  "prompt_medium_3", 
  "prompt_medium_3_reasoning"
]

parser = argparse.ArgumentParser(description='tokenizer')
parser.add_argument('--prompt', "-p", type=str, help="prompt name", default=None, nargs="*")

def gemini_request(client, model_name, prompt, batch):
  return client.generate_content(model_name=model_name, prompt=prompt, batch=batch)

def openai_request(client, model_name, prompt, batch):
  if model_name[:2] == "o1":
    messages = [{"role": "user", "content": prompt + "\n\n" + "\n".join(batch)}]
  else:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n".join(batch)}
    ]
  completion = client.chat.completions.create(
        model=model_name,
        seed=0,
        messages=messages
      )
  return completion.choices[0].message.content

def hf_request(client, model_name, prompt, batch):
  messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": "\n".join(batch)}
  ]
  return client(messages)

if __name__ == "__main__":
  args = parser.parse_args()
  dataset = "confiant"
  split = "test"
  path_to_bitexts = Path(f"data/{dataset}/bitexts")
  path_to_prompts = Path("prompts")
  path_to_result = Path(f"results/{dataset}/{split}")
  test_examples, = read_data([path_to_bitexts / f"{split}.fra"])
  model_name = "gemini-1.5-pro-001"

  batch_size = 1
  total_size = len(test_examples)
  num_batches = total_size // batch_size + int(total_size % batch_size != 0)
  prompts = []
  for promptfile in path_to_prompts.glob("*.txt"):
    name = promptfile.stem
    if (args.prompt is not None and name in args.prompt) or (args.prompt is None and name not in PROMPTS_TO_EXCLUDE):
      with open(promptfile, "r", encoding="utf-8") as fp:
        prompts.append((promptfile.stem, fp.read()))
  
  model_path_name = "_".join(model_name.split(":")[1:]) if ':' in model_name else model_name
  path_out = path_to_result / (model_path_name)
  path_out.mkdir(parents=True, exist_ok=True)

  if model_name[:9] == "mistralai":
    client = pipeline("text-generation", model=model_name, max_new_tokens=4096)
    make_request = hf_request
  elif model_name[:6] == "gemini":
    client = GeminiAPI(os.getenv("GOOGLE_GENAI_KEY"))
    make_request = gemini_request
  else:
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"), organization=os.getenv("ORGANIZATION_ID"))
    make_request = openai_request
    
  for name, prompt in prompts:
    answers = []
    if model_name[:6] == "gemini":
      # cache prompt
      cache_name = client.cache_prompt(model_name, prompt, time=300)
    for i in range(num_batches):
      batch = test_examples[(i*batch_size):((i+1)*batch_size)]
      print(batch)
      res = make_request(client, model_name, prompt, batch)
      answers.append(res)

    assert len(answers) == len(test_examples)
    df = pd.DataFrame({"request":test_examples, "answer": answers})
    df["answer"] = df["answer"].str.replace("\n", "\\n")
    df["request"] = df["request"].str.replace("\n", "\\n")
    df.to_csv(path_out / f"answer_to_{name}.csv", index=False)