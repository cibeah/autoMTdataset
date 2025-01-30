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

| Model | Prompt          | chrF |  chrF++ | BLEU | BLEU (Kreyol Tokenizer) |
|-------|-----------------|-------|------|----|----|
|  gpt-3 (fine-tuned on gcf-tatoeba)  | prompt_en     | 23.72   |  23.07    | 4.85 | 5.07 |
|       | prompt_fr | 24.97   |   24.16   | 4.4 | 5.06 |
|       | prompt_test | 23.06  |   22.61   | 4.83 | 5.2 |
| gpt-3  (fine-tuned on mart-confiant)  | prompt_en      | 29.81  |   29.36   | 9.54 | 9.66 |
|       | prompt_fr | 29.78  |  29.3   | 9.91 | 9.8 |
| gpt-4-1106-preview   | prompt_en     | **39.5**  |  38.56    | **14.41** | 16.22 |
|       | prompt_fr | **39.49**  |   **38.61**   | 14.06 | **16.48** |

Evaluation on `gcf-tatoeba`: evaluation dataset

| Model | Prompt          | chrF |  chrF++ | BLEU | BLEU (Kreyol Tokenizer) |
|-------|-----------------|-------|------|----|----|
|  gpt-3 (fine-tuned on gcf-tatoeba)  | prompt_en     | 28.73   |  28.65    | 8.8 | 7.75 |
|       | prompt_fr | 30.0   |   29.8   | 9.25 | 8.02 |
|       | prompt_test | 27.98  |   28.28   | 10.06 | 8.41 |
| gpt-3  (fine-tuned on mart-confiant)  | prompt_en      | 31.75  |   32.02   | 11.66 | 9.43 |
|       | prompt_fr | 29.48  |   29.58   | 9.89 | 8.54 |
| gpt-4-1106-preview   | prompt_en     | **43.27**  |  **43.09**    | **20.14** | **19.6** |
|       | prompt_fr | 42.03  |   41.85   | 17.29 | 17.57 |