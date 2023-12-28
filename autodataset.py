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
  path_to_bitexts = Path("data/bitexts")
  out_path = Path("data/data.jsonl")
  num_examples = 1000
  num_tests = 10
  fr, mq = read_data(path_to_bitexts)
  test_examples, test_answers = fr[num_examples:(num_examples+num_tests)], mq[num_examples:(num_examples+num_tests)]

  system_role = f"You are a professional translator speaking fluent Martinican Creole and French. " + \
  "When the user sends you a list of sentences in French, please simply return a list of the translations " + \
  "in Martinican creole."

  # Check for file on server
  client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
  file_id = None
  files = client.files.list()
  for file in files.data:
    if file.filename == "data.jsonl":
       file_id = file.id

  # Create and upload file on server if neessary
  if file_id is None:
    if not out_path.exists():
      train_data = create_jsonl(fr[:num_examples],mq[:num_examples], system_role, out_path) 
    # Check file validity
    with open(out_path, "r", encoding="utf-8") as fp:
      train_data = fp.read().split("\n")
    train_data = [json.loads(s) for s in train_data]
    format_error_checks(train_data)
    with open(out_path, "rb") as fp:
      res = client.files.create(
        file=fp,
        purpose="fine-tune"
      )
      file_id = res.id

  # Create fine tuning job
  client.fine_tuning.jobs.create(
    training_file=file_id, 
    model="gpt-3.5-turbo"
  )
  
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": system_role},
      {"role": "user", "content": "\n".join(test_examples)}
    ]
  )

  res = completion.choices[0].message.content.split("\n")
  df = pd.DataFrame(test_examples,res)
  df.to_csv("answer.csv")
  print("\n".join([test_examples[i] + " => " + res[i] for i in range(num_tests)]))
  print("\n".join([test_examples[i] + " => " + test_answers[i] for i in range(num_tests)]))

   # system_role = f"""You are a professional translator speaking fluent Martinican Creole and French. 
  # When the user sends you a list of sentences in French, please simply return a list of the translations
  # in Martinican creole. 
  # To illustrate the task, I am going to give you some examples of sentences in French, and then their translations in Martinican creole. 
  # The sentences will be organized in the following way: first the sentence in French, then an arrow => and then the sentence 
  # in Martinican creole. 
  # Here are the examples: \n {"\n".join(prompt_examples)}.
  # """