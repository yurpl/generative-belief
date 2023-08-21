import pandas as pd
import re
golds = pd.read_csv("/Users/john/Desktop/Towards-Generative-EFP/GenerativeEFP/gpt3/preds_syn_flan_flan.csv")
gold, preds_f = golds['gold'].to_list(), golds['pred'].to_list()

import spacy
nlp = spacy.load('en_core_web_sm')
def parse_tree(tree_str, parent="AUTHOR"):
    tuples = []
    pattern = r'\((\w+)\s+(\w+)((?:\s+\(.*?\))*)\)(?=\s*\(|\s*$)'

    for m in re.finditer(pattern, tree_str):
        child, value, subtree_str = m.groups()
        tuples.append((parent, child, value))
        if subtree_str:
            tuples.extend(parse_tree(subtree_str.strip(), child))

    return tuples
ls = []
counter = 0
for gold_doc, pred_doc in zip(gold, preds_f):
    pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
    gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src'))

    p = set(parse_tree(pred_doc))
    g = parse_tree(gold_doc)

    dis = (set(p) - set(g))
    if dis:
        print( g, dis)

    counter+= len(dis)
    # for i in dis:
    #     if 'nest' in i:
    #         counter += 1
    # counter += len(dis)
    # for gg in g:
    #     for i in dis:
    #         if gg[0] != i[0] and gg[1] == i[1] and gg[2] == i[2]:
    #             counter += 1    # for pp in p:
    #     for i in dis:
    #         if pp[0] != i[0] and pp[1] == i[1] and pp[2] == i[2]:
    #             counter += 1
    # for gg in g:
    #     for i in dis:
    #         if gg[0] == i[0] and gg[1] == i[1] and gg[2] != i[2]:
    #             print(gg, i)
    #             counter += 1
    # counter += len(dis)
    # for i in dis:
    #     if i[1] =='GEN' or  i[0] == "GEN":
    #         counter+=1
    # for pp in p:
    #     for gg in g:
    #         if pp[0] != gg[0] and pp[1] == gg[1] and pp[2] == gg[2]:
    #             counter += 1
    # for ii in p:
    #     for gg in g:
    #         if ii[0] == gg[0] and ii[1] == gg[1] and ii[2] != gg[2] and ii[2] != 'nest':
    #             counter += 1
    # counter += len(dis)

    # unique_second_elements = set(item[1] for item in dis)
    # #
    # sublists = [[item for item in dis if item[1] == second_element] for second_element in unique_second_elements]
    # if len(sublists) > 1:
    #     counter += 1
from collections import Counter
counts = Counter(ls)
print(counter)
# # Iterate over the counts
# for value, count in counts.items():
#     print(f"{value}: {count}")
golds = pd.read_csv("/Users/john/Desktop/Towards-Generative-EFP/GenerativeEFP/gpt3/fb_syn_flan_preds.csv")
gold, preds_f = golds['gold'].to_list(), golds['pred'].to_list()

import spacy
nlp = spacy.load('en_core_web_sm')
def parse_tree(tree_str, parent="AUTHOR"):
    tuples = []
    pattern = r'\((\w+)\s+(\w+)((?:\s+\(.*?\))*)\)(?=\s*\(|\s*$)'

    for m in re.finditer(pattern, tree_str):
        child, value, subtree_str = m.groups()
        tuples.append((parent, child, value))
        if subtree_str:
            tuples.extend(parse_tree(subtree_str.strip(), child))

    return tuples
ls = []
counter = 0
for gold_doc, pred_doc in zip(gold, preds_f):
    pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
    gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src'))

    p = set(parse_tree(pred_doc))
    g = parse_tree(gold_doc)
    dis = (set(p) - set(g))
    counter += len(dis)
    # for pp in p:
    #     for gg in g:
    #         if pp[0] != gg[0] and pp[1] == gg[1] and pp[2] == gg[2]:
    #             counter += 1
    # for i in dis:
    #     if i[1] =='GEN' or  i[0] == "GEN":
    #         counter+=1
    # for ii in p:
    #     for gg in g:
    #         if ii[0] == gg[0] and ii[1] == gg[1] and ii[2] != gg[2] and ii[2] != 'nest':
    #             counter += 1
    # counter += len(dis)

    unique_second_elements = set(item[1] for item in dis)
    #
    sublists = [[item for item in dis if item[1] == second_element] for second_element in unique_second_elements]
    if len(sublists) > 1:
        counter += 1
from collections import Counter
counts = Counter(ls)
print(counter)
# # Iterate over the counts
# for value, count in counts.items():
#     print(f"{value}: {count}")
