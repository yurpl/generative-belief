import jsonlines
import openai
import pandas as pd
import time
#stop=[")"]
#
# rate_limit_per_minute = 20
# delay = 60.0 / rate_limit_per_minute



base_prompt = "Instruction: Given a sentence, your task is to identify all events mentioned and determine their truth value. For each event, provide an output in the format (event truth_value). Use true if the event is definitively happening based on the sentence, ptrue if it possibly happened, false if it possibly did not happen, false if it definitively did not happen, and unknown if it's uncertain. When one event is mentioned by a source in the text, use the nest token to represent this of format (source nest (event truth_value)). \n"
train = pd.read_json('/Users/john/Desktop/Towards-Generative-EFP/GenerativeEFP/gpt3/FB-lin-1s-random/train_1.jsonl', lines=True)
test = pd.read_json('/Users/john/Desktop/Towards-Generative-EFP/GenerativeEFP/gpt3/FB-lin-1s-random/test_prepared.jsonl', lines=True)



def generate_prompt(row, examples):
    prompt = row['prompt'] + row['completion']
    return prompt

model_engine = "text-davinci-003"
examples = train['completion'].tolist()
few_shot_examples = base_prompt
# Train the model on the train set
for index, row in train.iterrows():

    few_shot_examples += generate_prompt(row, examples) + "\n"

with jsonlines.open("/Users/john/Desktop/Towards-Generative-EFP/GenerativeEFP/gpt3/FB-lin-1s-random/1random.jsonl", 'w') as writer:
    for index, row in test.iterrows():
        # time.sleep(delay)
        temp = few_shot_examples + row['prompt']
        completions = openai.Completion.create(
            engine=model_engine,
            prompt=temp,
            max_tokens=128,
            temperature=0.0
        )
        generated_completion = completions.choices[0].text.strip()
        print(generated_completion)
        writer.write({"prompt": row['prompt'], "completion": generated_completion})
