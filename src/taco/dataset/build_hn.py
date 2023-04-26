from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from multiprocessing import Pool
from taco.utils import SimpleTrainPreProcessor as TrainPreProcessor


def load_ranking(rank_file, relevance, n_sample, depth, shuffle_negs):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, _, p_0, _, _, _ = next(lines).strip().split()

        curr_q = q_0
        negatives = [] if p_0 in relevance[q_0] else [p_0]

        while True:
            try:
                q, _, p, _, _, _ = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[:depth]
                    if shuffle_negs:
                        random.shuffle(negatives)
                    yield curr_q, relevance[curr_q], negatives[:n_sample]
                    curr_q = q
                    negatives = [] if p in relevance[q] else [p]
                else:
                    if p not in relevance[q]:
                        negatives.append(p)
            except StopIteration:
                negatives = negatives[:depth]
                if shuffle_negs:
                    random.shuffle(negatives)
                yield curr_q, relevance[curr_q], negatives[:n_sample]
                return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, required=True)
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--hn_file', required=True)
    parser.add_argument('--qrels', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--template', type=str, default=None)

    parser.add_argument('--truncate', type=int, default=128)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--depth', type=int, default=200)
    parser.add_argument('--mp_chunk_size', type=int, default=500)
    parser.add_argument('--shard_size', type=int, default=45000)
    parser.add_argument('--shard_hn',action='store_true')
    parser.add_argument('--shuffle_negatives', action='store_true')

    args = parser.parse_args()
    random.seed(args.seed)

    qrel = TrainPreProcessor.read_qrel(args.qrels)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
                                              use_fast=True)
    processor = TrainPreProcessor(
        query_file=args.queries,
        collection_file=args.collection,
        tokenizer=tokenizer,
        max_length=args.truncate,
        template=args.template,
    )
    counter = 0
    shard_id = 0
    f = None
    os.makedirs(args.save_to, exist_ok=True)

    pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth,
                             args.shuffle_negatives))
    if args.shard_hn:
        with Pool() as p:
            for x in p.imap(processor.process_one, pbar,
                            chunksize=args.mp_chunk_size):
                counter += 1
                if f is None:
                    f = open(
                        os.path.join(args.save_to,
                                     f'split{shard_id:02d}.hn.jsonl'),
                        'w')
                    pbar.set_description(f'split - {shard_id:02d}')
                f.write(x + '\n')

                if counter == args.shard_size:
                    f.close()
                    f = None
                    shard_id += 1
                    counter = 0

        if f is not None:
            f.close()
    else:
        with open(os.path.join(args.save_to, 'train.hn.jsonl'), 'w') as f:
            with Pool() as p:
                for x in p.imap(processor.process_one, pbar,
                                chunksize=args.mp_chunk_size):
                    f.write(x + '\n')

