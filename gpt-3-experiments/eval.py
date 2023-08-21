n_gold_full = 0
n_pred_full = 0
n_tp_full = 0
import pandas as pd
import re
df = pd.read_csv('/Users/john/Desktop/Towards-Generative-EFP/GenerativeEFP/gpt3/fb_syn_flan_preds.csv', encoding= 'unicode_escape')
gold_pt, pred_pt = df['gold'].to_list(), df['pred'].to_list()
def parse_tree(tree_str, parent="AUTHOR"):
    tuples = []
    pattern = r'\((\w+)\s+(\w+)((?:\s+\(.*?\))*)\)(?=\s*\(|\s*$)'

    for m in re.finditer(pattern, tree_str):
        child, value, subtree_str = m.groups()
        tuples.append((parent, child, value))
        if subtree_str:
            tuples.extend(parse_tree(subtree_str.strip(), child))

    return tuples

for g_doc in gold_pt:
    g_doc = (g_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))
    tuples = (parse_tree(g_doc))

    tuples = (tuples)
    for i in tuples:
        n_gold_full += 1

for p_doc in pred_pt:
    p_doc = (p_doc.replace('(', ' (').replace('<src>', 'src'))

    tuples = (parse_tree(p_doc))
    tuples = set(tuples)

    for i in tuples:
        n_pred_full += 1

for gold_doc, pred_doc in zip(gold_pt, pred_pt):
    pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
    gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))

    p = parse_tree(pred_doc)
    p = set(p)
    g = parse_tree(gold_doc)
    g = set(g)
    for pp in p:
        for gg in g:
            if pp == gg:
                n_tp_full += 1

precision_full = float(n_tp_full) / float(n_pred_full) if n_pred_full != 0 else 0
recall_full = float(n_tp_full) / float(n_gold_full) if n_gold_full != 0 else 0
f1_full = 2 * precision_full * recall_full / (precision_full + recall_full) if precision_full != 0 or recall_full != 0 else 0

n_gold = 0
n_pred = 0
n_tp = 0
if '(' in pred_pt[0]:
    for g_doc in gold_pt:
        g_doc = (g_doc.replace('(', ' (').replace(' END', ''))

        tuples = (parse_tree(g_doc))
        for i in tuples:
            if i[0] == "AUTHOR":
                n_gold += 1

    p2 = []
    for p_doc in pred_pt:
        p_doc = (p_doc.replace('(', ' ('))

        tuples = (parse_tree(p_doc))
        tuples = set(tuples)
        for i in tuples:
            if i[0] == "AUTHOR":
                n_pred += 1

    for gold_doc, pred_doc in zip(gold_pt, pred_pt):
        pred_doc = (pred_doc.replace('(', ' ('))
        gold_doc = (gold_doc.replace('(', ' (').replace(' END', ''))

        p = set(parse_tree(pred_doc))
        g = parse_tree(gold_doc)
        for pp in p:
            for gg in g:
                if pp == gg and pp[0] == "AUTHOR" and gg[0] == "AUTHOR":
                    n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0


    ###########F1 for NAuthor

    n_gold_na = 0
    n_pred_na = 0
    n_tp_na = 0

    for g_doc in gold_pt:
        g_doc = (g_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))

        tuples = (parse_tree(g_doc))
        for i in tuples:
            if i[0] != "AUTHOR":
                n_gold_na += 1

    p2 = []
    for p_doc in pred_pt:
        p_doc = (p_doc.replace('(', ' (').replace('<src>', 'src'))

        tuples = (parse_tree(p_doc))
        tuples = set(tuples)
        for i in tuples:
            if i[0] != "AUTHOR":
                n_pred_na += 1

    for gold_doc, pred_doc in zip(gold_pt, pred_pt):
        pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
        gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))

        p = parse_tree(pred_doc)
        p = set(p)
        g = parse_tree(gold_doc)
        for pp in p:
            for gg in g:
                if pp == gg and pp[0] != "AUTHOR" and gg[0] != "AUTHOR":
                    n_tp_na += 1

    precision_na = float(n_tp_na) / float(n_pred_na) if n_pred_na != 0 else 0
    recall_na = float(n_tp_na) / float(n_gold_na) if n_gold_na != 0 else 0

    f1_na = 2 * precision_na * recall_na / (precision_na + recall_na) if precision_na != 0 or recall_na != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1,
              'precision_F': precision_full, 'recall_F': recall_full, 'f1_F': f1_full
              # 'precision_source': precision_s, 'recall_source': recall_s, 'f1_source': f1_s,
              # 'precision_target': precision_t, 'recall_target': recall_t, 'f1_target': f1_t,
              # 'precision_st': precision_st, 'recall_st': recall_st, 'f1_st': f1_st,
              # 'precision_f': precision_f, 'recall_f': recall_f, 'f1_fact': f1_f
              }
    print(scores)

    n_gold = 0
    n_pred = 0
    n_tp = 0

    for gold_doc, pred_doc in zip(gold_pt, pred_pt):
        pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
        gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))

        p = parse_tree(pred_doc)
        g = parse_tree(gold_doc)
        ct_p = []
        ct_gold = []

        for ct in p:
            if (ct[2]) == 'pfalse' and ct[0] == "AUTHOR":
                ct_p.append(ct)
        for ctx in g:
            if (ctx[2]) == 'pfalse' and ctx[0] == "AUTHOR":
                ct_gold.append(ctx)
        n_gold += len(ct_gold)
        n_pred += len(set(ct_p))
        for ii in ct_gold:
            for jj in ct_p:
                if ii == jj:
                    n_tp += 1
        precision_triplet = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall_triplet = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1_pr = 2 * precision_triplet * recall_triplet / (
                precision_triplet + recall_triplet) if precision_triplet != 0 or recall_triplet != 0 else 0

    n_gold = 0
    n_pred = 0
    n_tp = 0

    for gold_doc, pred_doc in zip(gold_pt, pred_pt):
        pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
        gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))

        p = parse_tree(pred_doc)
        g = parse_tree(gold_doc)
        ct_p = []
        ct_gold = []

        for ct in p:
            if (ct[2]) == 'ptrue' and ct[0] == "AUTHOR":
                ct_p.append(ct)
        for ctx in g:
            if (ctx[2]) == 'ptrue' and ctx[0] == "AUTHOR":
                ct_gold.append(ctx)
        n_gold += len((ct_gold))
        n_pred += len(set(ct_p))
        for ii in ct_gold:
            for jj in ct_p:
                if ii == jj:
                    n_tp += 1
        precision_triplet = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall_triplet = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1_pt = 2 * precision_triplet * recall_triplet / (
                precision_triplet + recall_triplet) if precision_triplet != 0 or recall_triplet != 0 else 0
    n_gold = 0
    n_pred = 0
    n_tp = 0
    for gold_doc, pred_doc in zip(gold_pt, pred_pt):
        pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
        gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src'))

        p = parse_tree(pred_doc)
        g = parse_tree(gold_doc)
        ct_p = []
        ct_gold = []

        for ct in p:
            if (ct[2]) == 'true' and ct[0] == "AUTHOR":
                ct_p.append(ct)
        for ctx in g:
            if (ctx[2]) == 'true' and ctx[0] == "AUTHOR":
                ct_gold.append(ctx)
        n_gold += len((ct_gold))
        n_pred += len(set(ct_p))
        for ii in ct_gold:
            for jj in ct_p:
                if ii == jj:
                    n_tp += 1
        precision_triplet = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall_triplet = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1_t = 2 * precision_triplet * recall_triplet / (
                precision_triplet + recall_triplet) if precision_triplet != 0 or recall_triplet != 0 else 0

    n_gold = 0
    n_pred = 0
    n_tp = 0
    for gold_doc, pred_doc in zip(gold_pt, pred_pt):
        pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
        gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))

        p = parse_tree(pred_doc)
        g = parse_tree(gold_doc)
        ct_p = []
        ct_gold = []

        for ct in p:
            if (ct[2]) == 'false' and ct[0] == "AUTHOR":
                ct_p.append(ct)
        for ctx in g:
            if (ctx[2]) == 'false' and ctx[0] == "AUTHOR":
                ct_gold.append(ctx)
        n_gold += len((ct_gold))
        n_pred += len(set(ct_p))
        for ii in ct_gold:
            for jj in ct_p:
                if ii == jj:
                    n_tp += 1
        precision_triplet = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall_triplet = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1_f = 2 * precision_triplet * recall_triplet / (
                precision_triplet + recall_triplet) if precision_triplet != 0 or recall_triplet != 0 else 0
    n_gold = 0
    n_pred = 0
    n_tp = 0
    for gold_doc, pred_doc in zip(gold_pt, pred_pt):
        pred_doc = (pred_doc.replace('(', ' (').replace('<src>', 'src'))
        gold_doc = (gold_doc.replace('(', ' (').replace('<src>', 'src').replace(' END', ''))

        p = parse_tree(pred_doc)
        g = parse_tree(gold_doc)
        ct_p = []
        ct_gold = []

        for ct in p:
            if (ct[2]) == 'unknown' and ct[0] == "AUTHOR":
                ct_p.append(ct)
        for ctx in g:
            if (ctx[2]) == 'unknown' and ctx[0] == "AUTHOR":
                ct_gold.append(ctx)
        n_gold += len((ct_gold))
        n_pred += len(set(ct_p))
        for ii in ct_gold:
            for jj in (ct_p):
                if ii == jj:
                    n_tp += 1
        precision_triplet = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall_triplet = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1_u = 2 * precision_triplet * recall_triplet / (
                precision_triplet + recall_triplet) if precision_triplet != 0 or recall_triplet != 0 else 0

    print({'macrof1': (f1_pr + f1_u + f1_t + f1_pt + f1_f) / 5})
    print({"PR-": f1_pr, "PR+": f1_pt, "CT+": f1_t, "UU": f1_u, "CT-": f1_f})