import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pathlib import Path

from utils.utils import read_data
from utils.validate import format_error_checks

load_dotenv()

def create_jsonl(fr,mq, system_role, out_path):
  data = [
    {"messages": [
        {"role": "system", "content": system_role}, 
        {"role": "user", "content": sentence_fr}, 
        {"role": "assistant", "content": sentence_mq}
    ]}
  for sentence_fr, sentence_mq in zip(fr, mq)
  ]
  str_data = [json.dumps(s, ensure_ascii=False) for s in data]
  with open(out_path, "w", encoding="utf-8") as fp:
     fp.write("\n".join(str_data))
  return data


if __name__ == "__main__":
  path_to_bitexts = Path("data/confiant/bitexts")
  out_path = Path("data/confiant")
  prompt_path = Path("prompts/prompt_fr.txt")
  num_examples = 1000
  num_tests = 10
  
  with open(prompt_path, "r", encoding="utf-8") as fp:
    system_role = fp.read()

  # Check for file on server
  client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
  train_file_id = None
  files = client.files.list()
  for file in files.data:
    if file.filename == "train_data.jsonl":
       train_file_id = file.id
    if file.filename == "eval_data.jsonl":
       eval_file_id = file.id

  # Create and upload file on server if neessary
  if train_file_id is None:
    if not (out_path / "train_data.jsonl").exists():
      train_examples, train_answers = read_data([path_to_bitexts / "train.fra", path_to_bitexts / "train.mart"])
      eval_examples, eval_answers = read_data([path_to_bitexts / "eval.fra", path_to_bitexts / "eval.mart"])
      train_data = create_jsonl(train_examples, train_answers, system_role, out_path / "train_data.jsonl") 
      train_data = create_jsonl(eval_examples, eval_answers, system_role, out_path / "eval_data.jsonl") 
    # Check file validity
    with open(out_path / "train_data.jsonl", "r", encoding="utf-8") as fp:
      train_data = fp.read().split("\n")
    train_data = [json.loads(s) for s in train_data]
    format_error_checks(train_data)
    with open(out_path / "train_data.jsonl", "rb") as fp:
      res = client.files.create(
        file=fp,
        purpose="fine-tune"
      )
      train_file_id = res.id
    with open(out_path / "eval_data.jsonl", "rb") as fp:
      res = client.files.create(
        file=fp,
        purpose="fine-tune"
      )
      eval_file_id = res.id

  # Create fine tuning job
  client.fine_tuning.jobs.create(
    training_file=train_file_id, 
    validation_file=eval_file_id, 
    model="gpt-3.5-turbo"
  )