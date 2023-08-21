import openai
import pandas as pd
import time
import spacy
nlp = spacy.load("en_core_web_sm")
#stop=[")"]

rate_limit_per_minute = 20
delay = 60.0 / rate_limit_per_minute

#openai.api_key = KEY
test = pd.read_json('/Users/john/Desktop/Towards-Generative-EFP/GenerativeEFP/gpt3/modal_lin_synflan/dev_prepared.jsonl', lines=True)
results = []
ft_model = 'davinci:ft-john-murzaku:mdp-flansyn-2023-05-11-22-28-39'

for i, j in zip(test['prompt'], test['completion']):
    res = openai.Completion.create(model=ft_model, prompt=i, temperature=0.0, max_tokens=1100, stop=[" END"])
    results.append(res['choices'][0]['text'])
    print(res['choices'][0]['text'])
results = pd.DataFrame(results)
results.to_csv("mdp_syn_flan_preds_dev.csv", index=False, header=['pred'])