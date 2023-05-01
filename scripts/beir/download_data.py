from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=True, default='',
                    help='dataset name from BEIR')
parser.add_argument('--out_dir', type=str, help='beir output dir')

args = parser.parse_args()
dataset_name = args.dataset_name

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
    dataset_name)
data_path = util.download_and_unzip(url, args.out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
    split="test")
print([len(corpus), len(queries), len(qrels)])
os.remove(os.path.join(args.out_dir, f'{dataset_name}.zip'))
# os.remove('datasets/' + dataset_name + '.zip')
