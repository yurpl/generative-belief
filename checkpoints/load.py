import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import spacy

nlp = spacy.load("en_core_web_sm")

model = T5ForConditionalGeneration.from_pretrained("PATH_TO_CHECKPOINT")
tokenizer = T5Tokenizer.from_pretrained("PATH_TO_TOKENIZER")

device = torch.device(f'cuda:{5}')  # Use the appropriate GPU index
model.to(device)
model.eval()

max_seq_length = 100  # Define your max sequence length here

def extract_parentheses_content(s):
    stack = []
    contents = []
    start = 0

    for i, char in enumerate(s):
        if char == '(':
            if not stack:
                start = i + 1
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
                if not stack:
                    contents.append(s[start:i])
    return contents

def tree2triplet(tree):
    triplets = []
    contents = extract_parentheses_content(tree)

    for values in contents:
        if 'nest' not in values:
            values = values.split(' ')
            triplets.append(
                ('AUTHOR', values[0], values[-1])
            )
        else:
            top_source = values[:values.find("nest") - 1]
            values = values[values.find("nest") + len("nest"):]
            values_nest = extract_parentheses_content(values)
            for nest_v in values_nest:
                if 'nest' not in nest_v:
                    nest_v = nest_v.split(' ')
                    triplets.append(
                        (f'AUTHOR_{top_source}', nest_v[0], nest_v[-1])
                    )
                else:
                    top_nest_source = nest_v[:nest_v.find("nest") - 1]
                    nest_v = nest_v[nest_v.find("nest") + len("nest"):]
                    values_nest_nest = extract_parentheses_content(nest_v)
                    for nest_v_nest in values_nest_nest:
                        nest_v_nest = nest_v_nest.split(' ')
                        triplets.append(
                            (f'AUTHOR_{top_source}_{top_nest_source}', nest_v_nest[0], nest_v_nest[-1])
                        )
    return triplets


with open('/home/jmurzaku/flant5/checkpoints/filtered_negative_case_timeslice2.jsonl', 'r') as f:
    lines = f.read().splitlines()

outputs, targets = [], []
for line in lines:
    data = json.loads(line)
    text = data['translatedContentText']

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    for sentence in sentences:
        tokenized_input = tokenizer.encode_plus(
            sentence,
            max_length=max_seq_length,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt"
        )

        outs = model.generate(
            input_ids=tokenized_input["input_ids"].to(device),
            attention_mask=tokenized_input["attention_mask"].to(device),
            max_length=max_seq_length
        )

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in tokenized_input["input_ids"]]
        print(tree2triplet(dec[0]))
        outputs.extend(dec)
        targets.extend(target)
