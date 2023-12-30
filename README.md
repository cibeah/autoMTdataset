Experiments: using LLM to create MT datasets

- [x] Add evaluation script
- [x] Directly split bitexts in train(/dev?)/test
- [ ] Prompt exploration
- [ ] Comparison with best prompt + finetuning
- [ ] Comparison with NMT (from scratch, multilingual, transfer from mBART ?)


```
train_test_split(["data/confiant/bitexts/data.fra", "data/confiant/bitexts/data.mart"], train=0.8, eval=0.1, test=0.1)
```


### Results

Evaluation on `mart-confiant`: evaluation dataset

| Model | Prompt          | chrF2 | BLEU |
|-------|-----------------|-------|------|
|  gpt-3 (fine-tuned on gcf-tatoeba)  | prompt_en     | 27.93   |      |
|       | prompt_fr | 16.83   |      |
|       | prompt_test | 23.67  |      |
| gpt-3  (fine-tuned on mart-confiant)  | prompt_en      | 21.42  |      |
|       | prompt_fr | **31.23**  |      |
| gpt-4-1106-preview   | prompt_en     | 28.68  |      |
|       | prompt_fr | 28.61  |      |

Evaluation on `gcf-tatoeba`: evaluation dataset

| Model | Prompt          | chrF2 | BLEU |
|-------|-----------------|-------|------|
|  gpt-3 (fine-tuned on gcf-tatoeba)  | prompt_en     | 20.41   |      |
|       | prompt_fr | 37.61   |      |
|       | prompt_test | 25.55  |      |
| gpt-3  (fine-tuned on mart-confiant)  | prompt_en      | 50.56  |      |
|       | prompt_fr | 22.56  |      |
| gpt-4-1106-preview   | prompt_en     | **100.0**  |      |
|       | prompt_fr | **100.0** |      |