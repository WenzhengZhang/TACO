import torch
import random
import bisect
from typing import Dict, List
import csv
import json
import datasets
from transformers import PreTrainedTokenizer
import logging
import unicodedata
import warnings
from dataclasses import dataclass
import regex
from tqdm import tqdm
import numpy as np
import math

logger = logging.getLogger()


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def sample_range_excluding(n, k, excluding):
    skips = [j - i for i, j in enumerate(sorted(set(excluding)))]
    s = random.sample(range(n - len(skips)), k)
    return [i + bisect.bisect_right(skips, i) for i in s]


def get_idx(obj):
    example_id = obj.get("_id", None)
    if example_id is None:
        example_id = obj.get("id", None)
    if example_id is None:
        example_id = obj.get("text_id", None)
    if example_id is None:
        raise ValueError(
            "No id field found in data, tried `_id`, `id`, `text_id`")
    example_id = str(example_id) if example_id is not None else None
    return example_id


def find_all_markers(template: str):
    """
    Find all markers' names (quoted in "<>") in a template.
    """
    markers = []
    start = 0
    while True:
        start = template.find("<", start)
        if start == -1:
            break
        end = template.find(">", start)
        if end == -1:
            break
        markers.append(template[start + 1:end])
        start = end + 1
    return markers


def fill_template(template: str, data: Dict, markers: List[str] = None,
                  allow_not_found: bool = False):
    """
    Fill a template with data.
    """
    if markers is None:
        markers = find_all_markers(template)
    for marker in markers:
        marker_hierarchy = marker.split(".")
        found = True
        content = data
        for marker_level in marker_hierarchy:
            content = content.get(marker_level, None)
            if content is None:
                found = False
                break
        if not found:
            if allow_not_found:
                warnings.warn(
                    "Marker '{}' not found in data. Replacing it with an empty string.".format(
                        marker), RuntimeWarning)
                content = ""
            else:
                raise ValueError(
                    "Cannot find the marker '{}' in the data".format(marker))
        template = template.replace("<{}>".format(marker), str(content))
    return template


@dataclass
class SimpleTrainPreProcessor:
    query_file: str
    collection_file: str
    columns = ['text_id', 'title', 'text']
    tokenizer: PreTrainedTokenizer

    max_length: int = 128
    title_field = 'title'
    text_field = 'text'
    template: str = None
    add_rand_negs: bool = False
    num_rands: int = 32
    use_doc_id_map: bool = False

    def __post_init__(self):
        self.queries = self.read_queries(self.query_file)
        self.collection = datasets.load_dataset(
            'csv',
            data_files=self.collection_file,
            column_names=self.columns,
            delimiter='\t',
            keep_default_na=False
        )['train']
        if self.use_doc_id_map:
            self.doc_id_map = self.get_doc_id_map()

    @staticmethod
    def read_queries(queries):
        qmap = {}
        with open(queries) as f:
            for l in f:
                qid, qry = l.strip().split('\t')
                qmap[qid] = qry.replace('[SEP]', '</s>')
        return qmap

    @staticmethod
    def read_qrel(relevance_file):
        qrel = {}
        with open(relevance_file, encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
        return qrel

    def get_doc_id_map(self):
        return {get_idx(self.collection[index]): index for index in range(len(
            self.collection))}

    def get_query(self, q):
        query_encoded = self.tokenizer.encode(
            self.queries[q],
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        return query_encoded

    def get_passage(self, p, use_doc_id_map=False):
        pid = self.doc_id_map[p] if use_doc_id_map else int(p)
        entry = self.collection[pid]
        title = entry[self.title_field]
        title = "" if title is None else title
        body = entry[self.text_field]
        body = '' if body is None else body
        if self.template is None:
            content = title + self.tokenizer.sep_token + body
        elif title == "":
            content = self.template.replace("<text>", body)
        else:
            content = self.template.replace("<title>", title).replace("<text>",
                                                                      body)

        passage_encoded = self.tokenizer.encode(
            content,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )

        return passage_encoded

    def process_one(self, train):
        q, pp, nn = train
        train_example = {
            'query': self.get_query(q),
            'positives': [self.get_passage(p, self.use_doc_id_map) for p in pp],
            'negatives': [self.get_passage(n, self.use_doc_id_map) for n in nn],
        }
        if self.add_rand_negs:
            if self.use_doc_id_map:
                pp = set([self.doc_id_map[p] for p in pp])
                nn = set([self.doc_id_map[n] for n in nn])
            else:
                pp = set([int(p) for p in pp])
                nn = set([int(n) for n in nn])
            rand_negs = sample_range_excluding(
                len(self.collection), self.num_rands, pp.union(nn))
            train_example['random_negatives'] = [self.get_passage(n, False)
                                                 for n in rand_negs]

        return json.dumps(train_example)


@dataclass
class SimpleCollectionPreProcessor:
    tokenizer: PreTrainedTokenizer
    separator: str = '\t'
    max_length: int = 128

    def process_line(self, line: str):
        xx = line.strip().split(self.separator)
        text_id, text = xx[0], xx[1:]
        text_encoded = self.tokenizer.encode(
            self.tokenizer.sep_token.join(text),
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        encoded = {
            'text_id': text_id,
            'text': text_encoded
        }
        return json.dumps(encoded)


def save_as_trec(rank_result: Dict[str, Dict[str, float]],
                 output_path: str, run_id: str = "OpenMatch"):
    """
    Save the rank result as TREC format:
    <query_id> Q0 <doc_id> <rank> <score> <run_id>
    """
    with open(output_path, "w") as f:
        for qid in rank_result:
            # sort the results by score
            sorted_results = sorted(rank_result[qid].items(),
                                    key=lambda x: x[1], reverse=True)
            for i, (doc_id, score) in enumerate(sorted_results):
                f.write("{} Q0 {} {} {} {}\n".format(qid, doc_id, i + 1, score,
                                                     run_id))


def check_answer(passages, answers, doc_ids, tokenizer):
    """Search through all the top docs to see if they have any of the answers."""
    hits = []
    for i, doc_id in enumerate(doc_ids):
        text = passages[doc_id][0]
        hits.append(has_answer(answers, text, tokenizer))
    return hits


def has_answer(answers, text, tokenizer):
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """

    if text is None:
        logger.warning("no doc in db")
        return False

    text = _normalize(text)

    # Answer is a list of possible strings
    text = tokenizer.tokenize(text).words(uncased=True)

    for single_answer in answers:
        single_answer = _normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
    return False


class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def _normalize(text):
    return unicodedata.normalize('NFD', text)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]


def convert_trec_to_nq_retrieval(trec_file,
                                 query_data,
                                 psg_data,
                                 out_file,
                                 store_raw=False):
    retrieval = {}
    tokenizer = SimpleTokenizer()
    with open(trec_file) as f_in:
        for line in tqdm(f_in.readlines()):
            question_id, _, doc_id, _, score, _ = line.strip().split()
            question_id = int(question_id)
            question = query_data[question_id]['question']
            answers = query_data[question_id]['answers']
            if answers[0] == '"':
                answers = answers[1:-1].replace('""', '"')
            # answers = eval(answers)
            if not isinstance(answers, list):
                answers = eval(answers)
            ctx = psg_data[int(doc_id) - 1]
            if question_id not in retrieval:
                retrieval[question_id] = {'question': question,
                                          'answers': answers, 'contexts': []}
            # title = ctx['title']
            text = ctx['text']
            answer_exist = has_answer(answers, text, tokenizer)
            if store_raw:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id,
                     'score': score,
                     'text': ctx,
                     'has_answer': answer_exist}
                )
            else:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id, 'score': score,
                     'has_answer': answer_exist}
                )

    json.dump(retrieval, open(out_file, 'w'), indent=4)


def evaluate_retrieval(retrieval_file, topk):
    tokenizer = SimpleTokenizer()
    retrieval = json.load(open(retrieval_file))
    accuracy = {k: [] for k in topk}
    max_k = max(topk)

    for qid in tqdm(list(retrieval.keys())):
        answers = retrieval[qid]['answers']
        contexts = retrieval[qid]['contexts']
        has_ans_idx = max_k  # first index in contexts that has answers

        for idx, ctx in enumerate(contexts):
            if idx >= max_k:
                break
            if 'has_answer' in ctx:
                if ctx['has_answer']:
                    has_ans_idx = idx
                    break
            else:
                text = ctx['text'].split('\n')[1]  # [0] is title, [1] is text
                if has_answer(answers, text, tokenizer):
                    has_ans_idx = idx
                    break

        for k in topk:
            accuracy[k].append(0 if has_ans_idx >= k else 1)

    for k in topk:
        print(f'Top{k}\taccuracy: {np.mean(accuracy[k]):.4f}')
    return accuracy


def shuffle_cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_task_hps(input_str, num_tasks):
    task_hps = input_str.split(',')
    if len(task_hps) == 1 and num_tasks > 1:
        task_hps = [int(task_hps[0])] * num_tasks
    else:
        task_hps = [int(h) for h in task_hps]
    return task_hps


class MultiTaskDataLoader:

    def __init__(self, single_loaders, up_sample=True):
        self.single_loaders = single_loaders
        self.num_batches = [len(loader) for loader in self.single_loaders]
        # need this for hf trainer to compute num examples
        bsz_sum = sum(loader.batch_size for loader in single_loaders)
        # min_idx = np.argmin(self.num_batches)
        # max_idx = np.argmax(self.num_batches)
        # idx = max_idx if up_sample else min_idx
        # drop_last = single_loaders[idx].drop_last
        num_examples = sum(len(loader.dataset) for loader in single_loaders)
        # num_examples = len(
        #     single_loaders[idx].dataset) * bsz_sum / self.num_batches[
        #                    idx]
        # num_examples = math.floor(num_examples) if drop_last else math.ceil(
        #     num_examples)
        self.dataset = [None] * num_examples
        self.up_sample = up_sample
        self.batch_size_sum = bsz_sum

    def __len__(self):
        length = max(self.num_batches) if self.up_sample else min(
            self.num_batches)
        return length

    def __iter__(self):
        # output batch of different tasks, might have different batch size
        if self.up_sample:
            return iter(zip(*(shuffle_cycle(loader) if len(loader) < max(
                self.num_batches) else loader for loader in
                              self.single_loaders)))
        return iter(zip(*(loader for loader in self.single_loaders)))
