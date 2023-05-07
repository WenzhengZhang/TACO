import argparse
import os
import json
from datasets import load_dataset
import csv
from taco.utils import get_idx
from tqdm import tqdm
import sys

csv.field_size_limit(sys.maxsize)


def get_corpus_id_map(corpus_path, output_corpus_path):
    doc_id_map = {}
    with open(corpus_path, 'r') as fin:
        with open(output_corpus_path, 'w',
                  newline='') as fout:
            tsv_r = csv.reader(fin, delimiter="\t")
            tsv_w = csv.writer(fout, delimiter='\t')
            for i, line in tqdm(enumerate(tsv_r)):
                _id, title, text = line
                _id = _id.replace("\"", "")
                doc_id_map[_id] = i
                tsv_w.writerow([i, title, text])
    return doc_id_map


def map_qrels(input_path, output_path, doc_id_map, delimiter):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fo:
        reader = csv.reader(fin, delimiter=delimiter)
        writer = csv.writer(fo, delimiter=delimiter)
        for i, line in tqdm(enumerate(reader)):
            q_id, _, d_id, score = line
            writer.writerow([q_id, 0, doc_id_map[d_id], score])


def map_hn(input_path, output_path, doc_id_map):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fo:
        reader = csv.reader(fin, delimiter=" ")
        writer = csv.writer(fo, delimiter=" ")
        for i, line in tqdm(enumerate(reader)):
            q_0, k1, p_0, k2, k3, k4 = line
            writer.writerow([q_0, k1, doc_id_map[p_0], k2, k3, k4])


def main(args):
    input_corpus_path = os.path.join(args.input_dir, args.corpus_name)
    output_corpus_path = os.path.join(args.output_dir, args.corpus_name)
    print('get corpus id map and process corpus')
    doc_id_map = get_corpus_id_map(input_corpus_path, output_corpus_path)
    if args.corpus_name != args.test_corpus_name:
        input_test_corpus_path = os.path.join(args.input_dir,
                                              args.test_corpus_name)
        output_test_corpus_path = os.path.join(args.output_dir,
                                               args.test_corpus_name)
        test_doc_id_map = get_corpus_id_map(input_test_corpus_path,
                                            output_test_corpus_path)
    else:
        test_doc_id_map = doc_id_map
    if args.map_train:
        print('map train qrels')
        input_qrel_path = os.path.join(args.input_dir, "train.qrel.tsv")
        output_qrel_path = os.path.join(args.output_dir, "train.qrel.tsv")
        map_qrels(input_qrel_path, output_qrel_path, doc_id_map, "\t")
        print("map train qrels trec")
        input_qrel_path = os.path.join(args.input_dir, "train.qrel.trec")
        output_qrel_path = os.path.join(args.output_dir, "train.qrel.trec")
        map_qrels(input_qrel_path, output_qrel_path, doc_id_map, " ")
        print("map train bm25 qrels")
        input_qrel_path = os.path.join(args.input_dir, "train.bm25.txt")
        output_qrel_path = os.path.join(args.output_dir, "train.bm25.txt")
        map_hn(input_qrel_path, output_qrel_path, doc_id_map)
    if args.map_dev:
        print('map dev qrels')
        input_qrel_path = os.path.join(args.input_dir, "dev.qrel.tsv")
        output_qrel_path = os.path.join(args.output_dir, "dev.qrel.tsv")
        map_qrels(input_qrel_path, output_qrel_path, doc_id_map, "\t")
        print("map dev qrels trec")
        input_qrel_path = os.path.join(args.input_dir, "dev.qrel.trec")
        output_qrel_path = os.path.join(args.output_dir, "dev.qrel.trec")
        map_qrels(input_qrel_path, output_qrel_path, doc_id_map, " ")

    if args.map_test:
        print('map test qrels')
        input_qrel_path = os.path.join(args.input_dir, "test.qrel.tsv")
        output_qrel_path = os.path.join(args.output_dir, "test.qrel.tsv")
        map_qrels(input_qrel_path, output_qrel_path, test_doc_id_map, "\t")
        print("map test qrels trec")
        input_qrel_path = os.path.join(args.input_dir, "test.qrel.trec")
        output_qrel_path = os.path.join(args.output_dir, "test.qrel.trec")
        map_qrels(input_qrel_path, output_qrel_path, test_doc_id_map, " ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--corpus_name', type=str, default='psg_corpus.tsv')
    parser.add_argument('--test_corpus_name', type=str,
                        default='psg_corpus.tsv')
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--map_train', action='store_true')
    parser.add_argument('--map_dev', action='store_true')
    parser.add_argument('--map_test', action='store_true')
    args = parser.parse_args()
    main(args)
