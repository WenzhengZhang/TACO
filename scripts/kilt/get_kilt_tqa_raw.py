# adapted from https://github.com/facebookresearch/KILT/blob/main/scripts/get_triviaqa_input.py

import sys
import requests
import tarfile
import os
import json
import argparse

from tqdm.auto import tqdm


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def store_data(filename, data):
    with open(filename, "w+") as outfile:
        for idx, element in enumerate(data):
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
            json.dump(element, outfile)
            outfile.write("\n")


parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, help='base directory')
args = parser.parse_args()

url = "http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"
tar_filename = "triviaqa-rc.tar.gz"
trivia_path = "triviaqa-rc/"
members = [
    "qa/wikipedia-dev.json",
]
# base = "kilt-data/retriever/tqa/"
input_files = [
    args.base + "triviaqa-dev_id-kilt.jsonl",
]
output_files = [
    args.base + "tqa-dev-kilt.jsonl",
]


def decompress(tar_file, path, members=None):
    """
    Extracts `tar_file` and puts the `members` to `path`.
    If members is None, all members on `tar_file` will be extracted.
    """
    tar = tarfile.open(tar_file, mode="r:gz")
    if members is None:
        members = tar.getmembers()
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        tar.extract(member, path=path)
        # set the progress description of the progress bar
        progress.set_description(f"Extracting {str(member)}")
    # or use this
    # tar.extractall(members=members, path=path)
    # close the file
    tar.close()


print("1. download TriviaQA original tar.gz file")
# Streaming, so we can iterate over the response.
r = requests.get(url, stream=True)
# Total size in bytes.
total_size = int(r.headers.get("content-length", 0))
block_size = 1024  # 1 Kibibyte
t = tqdm(total=total_size, unit="iB", unit_scale=True)
with open(tar_filename, "wb") as f:
    for data in r.iter_content(block_size):
        t.update(len(data))
        f.write(data)
t.close()
if total_size != 0 and t.n != total_size:
    print("ERROR, something went wrong")

print("2. extract tar.gz file")
decompress(tar_filename, trivia_path, members=members)

print("3. remove tar.gz file")
os.remove(tar_filename)

print("4. getting original questions")
id2input = {}
for member in members:
    print(member)
    filename = trivia_path + member
    with open(filename, "r") as fin:
        data = json.load(fin)
        for element in data["Data"]:
            e_id = element["QuestionId"]
            e_input = element["Question"]
            assert e_id not in id2input
            id2input[e_id] = e_input
    os.remove(filename)

print("5. remove original TriviaQA data")
os.rmdir(trivia_path + "qa/")
os.rmdir(trivia_path)

print("6. update kilt files")
for in_file, out_file in zip(input_files, output_files):
    data = load_data(in_file)
    for element in data:
        element["input"] = id2input[element["id"]]
    store_data(out_file, data)
    os.remove(in_file)
