import jsonlines
import csv
import pandas as pd
import argparse
import os
import re
import json
from datasets import load_dataset
from taco.utils import get_idx
import sys

csv.field_size_limit(sys.maxsize)


def process_qrel(input_dir, processed_dir, data_name, split, doc_id_map):
    qrel_path = os.path.join(input_dir, f"{data_name}/qrels/{split}.tsv")
    processed_qrel_path = os.path.join(processed_dir,
                                       f"{split}.qrel.tsv")
    qrel_trec_path = os.path.join(processed_dir, f"{split}.qrel.trec")
    q_ids = []
    qrel_df = pd.read_csv(qrel_path, sep='\t')
    with open(processed_qrel_path, 'w', newline='') as fout:
        tsv_w = csv.writer(fout, delimiter='\t')
        for i in range(len(qrel_df)):
            score = qrel_df.loc[i, 'score']
            q_id = qrel_df.loc[i, 'query-id']
            d_id = qrel_df.loc[i, 'corpus-id'].replace("\"", "")
            doc_id = doc_id_map[d_id]
            # assert (qrel_df.loc[i, 'score'] == 1)
            tsv_w.writerow([q_id, 0, doc_id, score])
            q_ids.append(str(q_id))
    # Test Qrels TREC
    with open(qrel_trec_path, 'w') as fout:
        for line in open(processed_qrel_path):
            fout.write(line.replace('\t', ' '))

    q_ids = list(set(q_ids))
    return q_ids


def process_query(input_dir, processed_dir, data_name, q_ids, split):
    # Queries
    que_count = 0
    input_query_path = os.path.join(input_dir, f"{data_name}/queries.jsonl")
    processed_query_path = os.path.join(processed_dir,
                                        f"{split}.query.txt")
    with open(input_query_path, 'r', encoding='utf-8') as fin:
        with open(processed_query_path, 'w', newline='') as fout:
            tsv_w = csv.writer(fout, delimiter='\t')
            for item in jsonlines.Reader(fin):
                _id = item['_id']
                text = item['text']
                if str(_id) in q_ids:
                    tsv_w.writerow([_id, text])
                    que_count += 1
    print(f'queries count for {data_name}_{split} : {que_count}')


def process_corpus(input_dir, processed_dir, data_name):
    # Corpus
    doc_count = 0
    doc_id_map = {}
    input_corpus_path = os.path.join(input_dir,
                                     f'{data_name}/corpus.jsonl')
    output_corpus_path = os.path.join(processed_dir,
                                      'psg_corpus.tsv')
    with open(input_corpus_path, 'r') as fin:
        with open(output_corpus_path, 'w',
                  newline='') as fout:
            tsv_w = csv.writer(fout, delimiter='\t')
            for i, line in enumerate(fin):
                item = json.loads(line)
                _id = item['_id']
                _id = _id.replace("\"", "")
                doc_id_map[_id] = i
                title = item['title']
                if dataset_name == 'robust04':
                    text = re.sub(r"[^A-Za-z0-9=(),!?\'\`]", " ", item['text'])
                    text = " ".join(text.split())
                else:
                    text = item['text'].replace("\n", " ")
                tsv_w.writerow([i, title, text])
                doc_count += 1
    print(f'doc count {doc_count}')
    return doc_id_map


# TODO: filter nan lines
def check_corpus(processed_dir, cache_dir):
    corpus_path = os.path.join(processed_dir, 'psg_corpus.tsv')
    corpus_set = load_dataset(
        "csv",
        data_files=corpus_path,
        column_names=["id", "title", "text"],
        delimiter='\t',
        cache_dir=cache_dir
    )["train"]
    for example in corpus_set:
        try:
            example_id = get_idx(example)
        except ValueError:
            print(example)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, default='',
                        help='dataset name from BEIR')
    parser.add_argument('--input_dir', type=str, help='input directory')
    parser.add_argument('--processed_dir', type=str, help='processed directory')
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--process_train', action='store_true')
    parser.add_argument('--process_dev', action='store_true')
    parser.add_argument('--verify_corpus', action='store_true')
    parser.add_argument('--not_process_test', action='store_true')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    print('process corpus ... ')
    did_map = process_corpus(args.input_dir, args.processed_dir, dataset_name)
    if not args.not_process_test:
        print('process qrels test')
        test_qids = process_qrel(args.input_dir, args.processed_dir,
                                 dataset_name,
                                 'test', did_map)
        print('process queries test ... ')
        process_query(args.input_dir, args.processed_dir, dataset_name,
                      test_qids, 'test')
    if args.process_train:
        print('process qrels train ... ')
        train_qids = process_qrel(args.input_dir, args.processed_dir,
                                  dataset_name, 'train', did_map)
        print('process queries train ... ')
        process_query(args.input_dir, args.processed_dir, dataset_name,
                      train_qids, 'train')
    if args.process_dev:
        print('process qrels dev ... ')
        dev_qids = process_qrel(args.input_dir, args.processed_dir,
                                dataset_name, 'dev', did_map)
        print('process queries dev ... ')
        process_query(args.input_dir, args.processed_dir, dataset_name,
                      dev_qids, 'dev')
    # if args.verify_corpus:
    #     check_corpus(args.processed_dir, args.cache_dir)
