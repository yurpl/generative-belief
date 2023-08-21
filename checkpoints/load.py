import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import spacy

nlp = spacy.load("en_core_web_sm")

model = T5ForConditionalGeneration.from_pretrained("/home/jmurzaku/checkpoints")
tokenizer = T5Tokenizer.from_pretrained("/home/jmurzaku/checkpoints/tokenizer")

device = torch.device(f'cuda:{5}')  # Use the appropriate GPU index
model.to(device)
model.eval()

max_seq_length = 100  # Define your max sequence length here
PATH = 'PATH_TO_FILE'
with open(PATH, 'r') as f:
    lines = f.read().splitlines()

outputs, targets = [], []
for line in lines:
    data = json.loads(line)
    text = data['translatedContentText']

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    for sentence in sentences:
        tokenized_input = tokenizer.encode_plus(
            "tree to sentence: " + sentence,
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
        outputs.extend(dec)
        targets.extend(target)
