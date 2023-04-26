import json
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--template', type=str)
parser.add_argument('--tokenizer', type=str, required=False,
                    default='bert-base-uncased')
parser.add_argument('--minimum-negatives', type=int, required=False, default=1)
parser.add_argument('--max_length', type=int, required=True, default=256)
parser.add_argument('--max_q_len', type=int, required=True, default=32)
args = parser.parse_args()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                               use_fast=True)
with open(args.input) as f:
    data = json.load(f)
# data = load_dataset('json', data_files=args.input)['train']

template = args.template
if template is None:
    template = "<title> [SEP] <text>"

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(args.output, 'w') as f:
    for idx, item in enumerate(tqdm(data)):
        if len(item['hard_negative_ctxs']) < args.minimum_negatives or len(
                item['positive_ctxs']) < 1:
            continue

        group = {}
        positives = [template.replace("<title>", pos["title"]).replace("<text>",
                                                                       pos[
                                                                           "text"])
                     for pos in item['positive_ctxs']]
        negatives = [template.replace("<title>", neg["title"]).replace("<text>",
                                                                       neg[
                                                                           "text"])
                     for neg in item['hard_negative_ctxs']]

        query = tokenizer.encode(item['question'].replace('[SEP]', '</s>'),
                                 add_special_tokens=False,
                                 max_length=args.max_q_len, truncation=True)
        positives = tokenizer(
            positives, add_special_tokens=False, max_length=args.max_length,
            truncation=True, padding=False)['input_ids']
        negatives = tokenizer(
            negatives, add_special_tokens=False, max_length=args.max_length,
            truncation=True, padding=False)['input_ids']

        group['query'] = query
        group['positives'] = positives
        group['negatives'] = negatives

        f.write(json.dumps(group) + '\n')
