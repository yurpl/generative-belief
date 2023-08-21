# This file contains all data loading and transformation functions

import time

import pandas as pd
from torch.utils.data import Dataset
import ast

senttag2word = {

    'source': 'source',
    'target': 'target',
    'CT+': '3',
    'Uu': '0',
    'PR+': '2',
    'PS+': '1',
    'PS-': '-1',
    'PR-': '-2',
    'CT-': '-3',
    'NA': '0',
    'other': '0',
    'CTu': '0',
    'ROB': '0',


}


def combine(datasets):
    datasets = []


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    df = pd.read_csv(data_path)
    sents, labels = df['input_text'].to_list(), df['target_text']
    sents = [i.split() for i in sents]
    print(f"Total examples = {len(sents)}")
    return sents, labels


def get_annotated_uabsa_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        if tuples != []:
            # tup: ([3, 4], POS)
            for tup in tuples:
                ap, sent = tup[0], tup[1]
                if len(ap) == 1:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{senttag2word[sent]}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{senttag2word[sent]}]"
        annotated_targets.append(sents[i])

    return annotated_targets


def get_annotated_aope_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([3, 4], [2])
        for tup in tuples:
            ap, op = tup[0], tup[1]
            opt = [sents[i][j] for j in op]
            # multiple OT for one AP
            if '[' in sents[i][ap[0]]:
                if len(ap) == 1:
                    sents[i][ap[0]] = f"{sents[i][ap[0]][:-1]}, {' '.join(opt)}]"
                else:
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]][:-1]}, {' '.join(opt)}]"
            else:
                annotation = f"{' '.join(opt)}"
                if len(ap) == 1:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{annotation}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{annotation}]"
        annotated_targets.append(sents[i])

    return annotated_targets


def get_annotated_aste_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([2], [5], 'NEG')
        for tup in tuples:
            ap, op, sent = tup[0], tup[1], tup[2]
            op = [sents[i][j] for j in op]
            # multiple OT for one AP
            if '[' in sents[i][ap[0]]:
                # print(i)
                if len(ap) == 1:
                    sents[i][ap[0]] = f"{sents[i][ap[0]][:-1]}, {' '.join(op)}]"
                else:
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]][:-1]}, {' '.join(op)}]"
            else:
                annotation = f"{senttag2word[sent]}|{' '.join(op)}"
                if len(ap) == 1:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{annotation}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{annotation}]"
        annotated_targets.append(sents[i])
    return annotated_targets


def get_annotated_tasd_targets(sents, labels):
    targets = []
    num_sents = len(sents)
    sents_str = [' '.join(s) for s in sents]
    for i in range(num_sents):
        s_str = sents_str[i]
        at_dict = {}
        for triplet in labels[i]:
            at, ac, polarity = triplet[0], triplet[1], triplet[2]
            if at in at_dict:
                at_dict[at][0].append(ac)
            else:
                at_dict[at] = [[ac], polarity]
        for at, ac_pol in at_dict.items():
            if len(ac_pol[0]) == 1:
                annotated_at = f"[{at}|{ac_pol[0][0]}|{ac_pol[1]}]"
            else:
                annotated_at = f"[{at}|{', '.join(ac_pol[0])}|{ac_pol[1]}]"
            if at != 'NULL':
                # print('at:', at, 'replaced_at:', annotated_at)
                s_str = s_str.replace(at, annotated_at)
            else:
                s_str += f" {annotated_at}"
        targets.append(s_str)
    return targets


def get_extraction_uabsa_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                if len(tri[0]) == 1:
                    a = sents[i][tri[0][0]]
                else:
                    start_idx, end_idx = tri[0][0], tri[0][-1]
                    a = ' '.join(sents[i][start_idx:end_idx + 1])
                c = senttag2word[tri[1]]
                all_tri.append((a, c))
            label_strs = ['(' + ', '.join(l) + ')' for l in all_tri]
            targets.append('; '.join(label_strs))
    return targets


def get_extraction_aope_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx + 1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx + 1])
            all_tri.append((a, b))
        label_strs = ['(' + ', '.join(l) + ')' for l in all_tri]
        targets.append('; '.join(label_strs))
    return targets


def get_extraction_tasd_targets(sents, labels):
    targets = []
    for label in labels:
        label_strs = ['(' + ', '.join(l) + ')' for l in label]
        target = '; '.join(label_strs)
        targets.append(target)
    return targets


def get_extraction_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx + 1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx + 1])
            c = senttag2word[tri[2]]
            all_tri.append((a, b, c))
        label_strs = ['(' + ', '.join(l) + ')' for l in all_tri]
        targets.append('; '.join(label_strs))
    return targets


def get_transformed_io(data):
    data = pd.read_csv(data)
    inputs = data['input_text'].to_list()
    targets = data['target_text'].to_list()


    return inputs, targets

class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, paradigm, task, max_len):
        # 'data/aste/rest16/train.txt'
        self.data_path = f'data/{task}/{data_dir}/{data_type}.csv'
        self.paradigm = paradigm
        self.task = task
        self.temp = False
        self.max_len = max_len
        self.tokenizer = tokenizer



        self.inputs = []
        self.inputs_st = []
        self.inputs_source = []

        self.segmented_inputs = []
        self.segmented_targets = []

        self.source_targets = []
        self.triplet = []
        self.sources = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        triplet_ids = self.triplet[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        triplet_mask = self.triplet[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask,
                # "src_target_ids": src_target_ids, "src_target_mask": src_target_mask,
                "triplet_ids": triplet_ids, "triplet_mask":triplet_mask}
                # "source_st_ids": source_st_ids, "src_st_mask": src_st_mask,
                # 'source_only_text_ids': source_only_text, 'source_only_text_mask': source_only_text_mask,
                # 'source_only_target_ids': source_only_target, 'source_only_target_mask': source_only_target_mask
                # }

    def _build_examples(self):



        inputs, targets = get_transformed_io(self.data_path)
        for i in range(len(inputs)):
            input = inputs[i]
            target = targets[i]

            if type(target) != float:

                tokenized_input = self.tokenizer.batch_encode_plus(
                    [input.replace("tree to sentence: ", "Given a sentence, your task is to identify all events mentioned and determine their truth value. For each event, provide an output in the format (event truth_value). Use true if the event is definitively happening based on the sentence, ptrue if it possibly happened, false if it possibly did not happen, false if it definitively did not happen, and unknown if it's uncertain. When one event is mentioned by a source in the text, use the nest token to represent this of format (source nest (event truth_value)): ")], max_length=self.max_len, pad_to_max_length=True, truncation=True,
                    return_tensors="pt",
                )

                tokenized_input_st = self.tokenizer.batch_encode_plus(
                    [input.replace('source target factuality: ', 'attribution tuple: ')], max_length=self.max_len, pad_to_max_length=True, truncation=True,
                    return_tensors="pt",
                )

                tokenized_input_s = self.tokenizer.batch_encode_plus(
                    [input.replace('source target factuality: ', 'sources: ')], max_length=self.max_len,
                    pad_to_max_length=True, truncation=True,
                    return_tensors="pt",
                )


                tokenized_triplet = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
                    return_tensors="pt"
                )



                self.inputs.append(tokenized_input)
                self.triplet.append(tokenized_triplet)
                self.inputs_st.append(tokenized_input_st)
                self.inputs_source.append(tokenized_input_s)







def write_results_to_log(log_file_path, best_test_result, args, dev_results, test_results, global_steps):
    """
    Record dev and test results to log file
    """
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} on {1} under {2} | {3:.4f} | ".format(
        args.task, args.dataset, args.paradigm, best_test_result
    )
    train_settings = "Train setting: bs={0}, lr={1}, num_epochs={2}".format(
        args.train_batch_size, args.learning_rate, args.num_train_epochs
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"

    metric_names = ['f1', 'precision', 'recall']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        for name in metric_names:
            name_step = f'{name}_{gstep}'
            results_str += f"{name:<8}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}"
            results_str += ' ' * 5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"

    with open(log_file_path, "a+") as f:
        f.write(log_str)