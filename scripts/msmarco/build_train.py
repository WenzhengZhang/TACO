from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
import json
from datetime import datetime
from multiprocessing import Pool

from taco.utils import SimpleTrainPreProcessor as TrainDataPreProcessor

OUTPUT_TEXT_FILE_NAME = "split"
OUTPUT_ENCODED_FILE_NAME = "encoded_split"

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--negative_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--template', type=str, default=None)
parser.add_argument('--use_title', type=bool, default=True)
parser.add_argument('--truncate', type=int, default=512)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--mp_chunk_size', type=int, default=5000)
parser.add_argument('--shard_size', type=int, default=50000)
parser.add_argument('--add_rand_negs', action='store_true')
parser.add_argument('--use_doc_id_map', action='store_true')
parser.add_argument('--shuffle_negatives', action='store_true')
parser.add_argument('--shard_hn', action='store_true')
parser.add_argument('--num_hards', type=int, default=30)
parser.add_argument('--num_rands', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

counter = 0
total_cntr = 0
shard_id = 0
encoded_data_file = None
os.makedirs(args.save_to, exist_ok=True)
random.seed(args.seed)
qrel = TrainDataPreProcessor.read_qrel(args.qrels)
print(
    f"build_train.py: loaded qrel file from {args.qrels}, containining {len(qrel)} entries")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainDataPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    max_length=args.truncate,
    template=args.template,
    add_rand_negs=args.add_rand_negs,
    num_rands=args.num_rands,
    use_doc_id_map=args.use_doc_id_map
)

with open(args.negative_file) as nf:
    buffer_encoded = []


    def read_negative_files_line(l):
        q, nn = l.strip().split('\t')
        nn = nn.split(',')
        if args.shuffle_negatives:
            random.shuffle(nn)
        return q, qrel[q], nn[:args.n_sample]


    pbar = tqdm(map(read_negative_files_line, nf))
    with Pool() as p:
        for x in p.imap(processor.process_one, pbar,
                        chunksize=args.mp_chunk_size):
            tokenized_example = x[0], x[1]
            counter += 1
            total_cntr += 1
            if encoded_data_file is None:
                encoded_data_file = open(os.path.join(args.save_to,
                                                      f'{OUTPUT_ENCODED_FILE_NAME}-{shard_id:02d}.jsonl'),
                                         'w')

            buffer_encoded.append(tokenized_example)

            if counter == args.shard_size:
                for x in buffer_encoded:
                    encoded_data_file.write(
                        x + '\n')  # save the encoded version
                buffer_encoded = []
                encoded_data_file.close()
                encoded_data_file = None
                shard_id += 1
                counter = 0
                pbar.set_description(
                    f"saving to {OUTPUT_TEXT_FILE_NAME}-{shard_id:02d}.jsonl")

# cleanup any stragglers
if encoded_data_file is not None:
    for x in buffer_encoded:
        encoded_data_file.write(x + '\n')  # save the encoded version
    encoded_data_file.close()

print(
    f"Done building train data files, processed {total_cntr} examples total across {shard_id + 1} shards")
