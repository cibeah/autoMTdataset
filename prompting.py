import os

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pathlib import Path

from utils.utils import read_data

load_dotenv()

if __name__ == "__main__":
  path_to_bitexts = Path("data/bitexts")
  out_path = Path("data/data.jsonl")
  path_to_prompts = Path("prompts")
  path_to_result = Path("results")
  num_examples = 100
  num_tests = 10
  fr, mq = read_data(path_to_bitexts)
  prompt_examples = [fr[i] + " => " + mq[i] for i in range(num_examples)]
  test_examples, test_answers = fr[num_examples:(num_examples+num_tests)], mq[num_examples:(num_examples+num_tests)]

  # system_role = f"""You are a professional translator speaking fluent Martinican Creole and French. 
  # When the user sends you a list of sentences in French, please simply return a list of the translations
  # in Martinican creole. 
  # To illustrate the task, I am going to give you some examples of sentences in French, and then their translations in Martinican creole. 
  # The sentences will be organized in the following way: first the sentence in French, then an arrow => and then the sentence 
  # in Martinican creole. 
  # Here are the examples: \n {"\n".join(prompt_examples)}.
  # """

  prompts = []
  for promptfile in path_to_prompts.glob("*.txt"):
     with open(promptfile, "r", encoding="utf-8") as fp:
        prompts.append((promptfile.stem, fp.read()))
  
  client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
  path_to_result.mkdir(exist_ok=True)
  for name, prompt in prompts:
    completion = client.chat.completions.create(
      model="gpt-4-1106-preview",
      seed=0,
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n".join(test_examples)}
      ]
    )
    res = completion.choices[0].message.content.split("\n")
    df = pd.DataFrame(test_examples,res)
    df.to_csv(path_to_result / f"answer_to_{name}.csv")