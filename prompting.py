import argparse
from multiprocessing import Pool
import os
import re
import time

from dotenv import load_dotenv
# import google.generativeai as genai
from openai import OpenAI
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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
parser.add_argument('--model', "-m", type=str, help="model name", default="gemini-1.5-pro-002")
parser.add_argument('--prompt', "-p", type=str, help="prompt name", default=["prompt_final"], nargs="*")

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

def write_answer(answers, references, path):
  df = pd.DataFrame({"request":references, "answer": answers})
  df["answer"] = df["answer"].str.replace("\n", "\\n")
  df["request"] = df["request"].str.replace("\n", "\\n")
  df.to_csv(path, index=False)

def write_unaligned_answer(answers, references, path):
  answers = pd.Series(answers).str.replace("\n", "\\n")
  answers.to_csv(str(path)+".mart", index=False, header=None, encoding="utf-8")
  references = pd.Series(references).str.replace("\n", "\\n")
  references.to_csv(str(path)+".fra", index=False, header=None, encoding="utf-8")

if __name__ == "__main__":
  args = parser.parse_args()
  dataset, split = "training-parallel-commoncrawl", "data"
  # dataset, split = "kreyolmt-public-mart", "eval"
  # dataset, split = "restog", "data"
  path_to_bitexts = Path(f"data/{dataset}/bitexts")
  path_to_prompts = Path("prompts")
  path_to_result = Path(f"results/{dataset}/{split}")
  test_examples, = read_data([path_to_bitexts / f"{split}.fra"])
  model_name = args.model

  mini_batch_size = 32
  write_every = 10
  request_time = 0
  num_workers = 10
  start_batch = 31
  total_size = len(test_examples)
  batch_size = mini_batch_size * num_workers
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
    def pool_request(mini_batch):
      return make_request(client, model_name, prompt, mini_batch)
    if model_name[:6] == "gemini":
      # cache prompt
      # caching set to 1h - to increase if generation takes more time !
      client.cache_prompt(model_name, prompt, time=int(24*3600)) #everything I can do in one week
    for i in tqdm(range(start_batch,num_batches)):
      batch = test_examples[(i*batch_size):((i+1)*batch_size)]
      def next_mini_batch():
        for j in range(num_workers):
          mini_batch = batch[(j*mini_batch_size):((j+1)*mini_batch_size)]
          if len(mini_batch) > 0:
            yield mini_batch
      # print(batch)
      start = time.time()
      # res = make_request(client, model_name, prompt, batch)
      with Pool(processes=num_workers) as pool:
        pool_res = pool.map(pool_request, next_mini_batch())
      request_time += (time.time() - start)
      for res in pool_res:
        if res == '':
          answer = ['' for _ in range(mini_batch_size)]
        else:
          answer = re.split(r'[\n]+', res)
        answers.extend(answer)
      # answers.append(res.spli)

      if i % write_every == 0:
        print(f"Average request time: {request_time/(i+1) :.3}s")
        first_batch_id = max(start_batch, i+1-write_every)
        answered = test_examples[(first_batch_id*batch_size):((i+1)*batch_size)]
        path_to_file = path_out / f"answer_to_{name}_{first_batch_id}"
        write_unaligned_answer(answers, answered, path_to_file)
        # assert len(answers) == len(answered)
        # path_to_df = path_out / f"answer_to_{name}_{first_batch_id}"
        # write_answer(answers, answered, path_to_file)
        print(f"Written to {path_to_file}")
        answers = []
