import json
from tqdm import tqdm
import argparse
from copy import deepcopy
import csv
import random


def filter_wow_dev(dev_wow_file, raw_dev_wow_file, output_wow_file):
    with open(dev_wow_file) as f:
        val_data = json.load(f)
    q2item = {}
    for d in val_data:
        q2item[d['question'].replace('\n', ' ')] = d
    new_items = []
    with open(raw_dev_wow_file) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            query = item['input'].replace('\n', ' ')
            new_items.append(q2item[query])
    print(len(new_items))
    with open(output_wow_file, 'w') as f:
        json.dump(new_items, f)


def psg_to_group_map(kilt_psg):
    psg2grp = {}
    grp_wiki = {}
    with open(kilt_psg) as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for i, line in tqdm(enumerate(reader)):
            if i == 0:
                continue
            pid, text, wiki_title, wiki_id, _, _ = line
            if wiki_id not in grp_wiki:
                grp_wiki[wiki_id] = [pid]
            else:
                grp_wiki[wiki_id].append(pid)
    for k, ps in grp_wiki.items():
        for p in ps:
            psg2grp[p] = ps
    return psg2grp


def process_aida_train(aida_file, aida_query, aida_qrel, out_aida_file,
                       psg2grp):
    # build aida_query.txt, aida_qrel.txt
    out_items = []
    cnts = []
    with open(aida_file) as f_in, open(aida_query, 'w') as f_query, \
            open(aida_qrel, 'w') as f_qrel:
        items = json.load(f_in)
        qrel_writer = csv.writer(f_qrel, delimiter='\t')
        for i, item in tqdm(enumerate(items)):
            text = item['question']
            m_start = text.find('START_ENT')
            left_window = ' '.join(text[:m_start].split(' ')[-64:])
            query_text = left_window + text[m_start:]
            f_query.write(str(i) + '\t' + query_text + '\n')
            cnt = 0
            for p in item['positive_ctxs']:
                pids = psg2grp[str(p['psg_id'])]
                cnt += len(pids)
                for pid in pids:
                    qrel_writer.writerow([str(i), '0', str(pid), '1'])
            cnts.append(cnt)
            new_item = deepcopy(item)
            new_item['question'] = query_text
            out_items.append(new_item)
    with open(out_aida_file, 'w') as f:
        json.dump(out_items, f)
    print(f'maximum num positives {max(cnts)}')
    print(f'average num positives {sum(cnts)/len(cnts)}')


def process_aida_dev(aida_file, aida_query):
    # make query surround mention, word length 200
    with open(aida_file) as f_in, open(aida_query, 'w') as f_query:
        writer = csv.writer(f_query, delimiter='\t')
        for i, line in tqdm(enumerate(f_in)):
            item = json.loads(line)
            left_ctxs = item['meta']['left_context']
            right_ctxs = item['meta']['right_context']
            mention = item['meta']['mention']
            left_window = ' '.join(left_ctxs.split(' ')[-64:])
            q_text = left_window + ' [START_ENT] ' + mention + ' [END_ENT] ' \
                                                               '' + right_ctxs
            writer.writerow(
                [str(i), q_text.replace('\t', ' ').replace('\n', ' ')])


def process_kilt_doc_corpus(kilt_kb_file, kilt_corpus):
    # max doc length 200
    with open(kilt_kb_file) as f_in, open(kilt_corpus, 'w') as f_o:
        for line in tqdm(f_in):
            item = json.loads(line)
            title = item['wikipedia_title']
            if len(title) == 0:
                title = ' '
            text = ' '.join(
                [t.strip('\n').replace('::::', ' ') for t in item['text'][:5]])
            text = ' '.join(text.split(' ')[:200])
            if len(text) == 0:
                text = ' '
            f_o.write(
                str(item['wikipedia_id']) + '\t' + title + '\t' + text + '\n')


def process_kilt_psg_corpus(psg_file, psg_corpus):
    with open(psg_file) as f_in, open(psg_corpus, 'w') as f_out:
        reader = csv.reader(f_in, delimiter="\t", quotechar='"')
        writer = csv.writer(f_out, delimiter="\t")
        for i, line in tqdm(enumerate(reader)):
            pid, text, wiki_title, wiki_id, _, _ = line
            text = text.replace('\n', ' ')
            if i == 0:
                continue
            writer.writerow([pid, wiki_title, text])


def generate_files(orig_file, query_file, qrel_file, is_dev=False):
    with open(orig_file) as f_in, open(query_file, 'w') as f_query, \
            open(qrel_file, 'w') as f_qrel:
        items = json.load(f_in)
        delimiter = ' ' if is_dev else '\t'
        qrel_writer = csv.writer(f_qrel, delimiter=delimiter)
        query_writer = csv.writer(f_query, delimiter='\t')
        for i, item in tqdm(enumerate(items)):
            qid = str(i)
            query_text = item['question'].replace('\n', ' ').replace('\t', ' ')
            assert len(query_text) > 0
            query_writer.writerow([qid, query_text])
            for pos in item['positive_ctxs']:
                qrel_writer.writerow([qid, '0', str(pos['psg_id']), '1'])


def down_sample(large_file, small_file):
    random.seed(42)
    with open(large_file) as fl, open(small_file, 'w') as fs:
        large_data = json.load(fl)
        assert len(large_data) > 100000
        small_data = random.sample(large_data, 100000)
        json.dump(small_data, fs)


def main(args):
    if args.process_trex:
        print('down-sample trex data')
        down_sample(args.train_trex_file, args.train_trex_small)
        print('generate trex data')
        generate_files(args.train_trex_small, args.train_trex_query,
                       args.train_trex_qrel, False)
        generate_files(args.dev_trex_file, args.dev_trex_query,
                       args.dev_trex_qrel, True)
    if args.process_zsre:
        print('down sample large dataset:  zsre')
        down_sample(args.train_zsre_file, args.train_zsre_small)
        print('generate zero-shot RE data')
        generate_files(args.train_zsre_small, args.train_zsre_query,
                       args.train_zsre_qrel, False)
        generate_files(args.dev_zsre_file, args.dev_zsre_query,
                       args.dev_zsre_qrel, True)
    if args.process_nq:
        print('generate nq data')
        generate_files(args.train_nq_file, args.train_nq_query,
                       args.train_nq_qrel,
                       False)
        generate_files(args.dev_nq_file, args.dev_nq_query, args.dev_nq_qrel,
                       True)
    if args.process_tqa:
        print('generate tqa  data')
        generate_files(args.train_tqa_file, args.train_tqa_query,
                       args.train_tqa_qrel, False)
        generate_files(args.dev_tqa_file, args.dev_tqa_query, args.dev_tqa_qrel,
                       True)
    if args.process_hopo:
        print('generate hopo data')
        generate_files(args.train_hopo_file, args.train_hopo_query,
                       args.train_hopo_qrel, False)
        generate_files(args.dev_hopo_file, args.dev_hopo_query,
                       args.dev_hopo_qrel,
                       True)
    if args.process_wow:
        print('filter wow dev data')
        filter_wow_dev(args.dev_wow_file, args.raw_dev_wow_file,
                       args.out_dev_wow_file)
        print('generate wow data')
        generate_files(args.train_wow_file, args.train_wow_query,
                       args.train_wow_qrel)
        generate_files(args.out_dev_wow_file, args.dev_wow_query,
                       args.dev_wow_qrel, True)
    if args.process_fever:
        print('generate fever data')
        generate_files(args.train_fever_file, args.train_fever_query,
                       args.train_fever_qrel)
        generate_files(args.dev_fever_file, args.dev_fever_query,
                       args.dev_fever_qrel, True)
    if args.process_aida:
        print('generate aida data')
        print('process aida dev ')
        process_aida_dev(args.dev_aida_file, args.dev_aida_query)
        print('process aida train')
        print('get psg2grp map')
        psg2grp = psg_to_group_map(args.kilt_psgs)
        process_aida_train(args.train_aida_file, args.train_aida_query,
                           args.train_aida_qrel, args.train_aida_out, psg2grp)
    if args.process_psgs:
        print('process kilt psg corpus')
        process_kilt_psg_corpus(args.kilt_psgs, args.kilt_psg_corpus)
    if args.process_docs:
        print('process kilt doc')
        process_kilt_doc_corpus(args.kilt_kb_file, args.kilt_doc_corpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_wow_file', type=str)
    parser.add_argument('--train_wow_query', type=str)
    parser.add_argument('--train_wow_qrel', type=str)
    parser.add_argument('--dev_wow_file', type=str)
    parser.add_argument('--raw_dev_wow_file', type=str)
    parser.add_argument('--out_dev_wow_file', type=str)
    parser.add_argument('--dev_wow_query', type=str)
    parser.add_argument('--dev_wow_qrel', type=str)
    parser.add_argument('--train_trex_file', type=str)
    parser.add_argument('--train_trex_query', type=str)
    parser.add_argument('--train_trex_qrel', type=str)
    parser.add_argument('--dev_trex_file', type=str)
    parser.add_argument('--dev_trex_query', type=str)
    parser.add_argument('--dev_trex_qrel', type=str)
    parser.add_argument('--train_fever_file', type=str)
    parser.add_argument('--train_fever_query', type=str)
    parser.add_argument('--train_fever_qrel', type=str)
    parser.add_argument('--dev_fever_file', type=str)
    parser.add_argument('--dev_fever_query', type=str)
    parser.add_argument('--dev_fever_qrel', type=str)
    parser.add_argument('--train_zsre_file', type=str)
    parser.add_argument('--train_zsre_query', type=str)
    parser.add_argument('--train_zsre_qrel', type=str)
    parser.add_argument('--dev_zsre_file', type=str)
    parser.add_argument('--dev_zsre_query', type=str)
    parser.add_argument('--dev_zsre_qrel', type=str)
    parser.add_argument('--train_aida_file', type=str)
    parser.add_argument('--train_aida_query', type=str)
    parser.add_argument('--train_aida_qrel', type=str)
    parser.add_argument('--dev_aida_file', type=str)
    parser.add_argument('--dev_aida_query', type=str)
    parser.add_argument('--dev_aida_qrel', type=str)
    parser.add_argument('--kilt_kb_file', type=str)
    parser.add_argument('--kilt_doc_corpus', type=str)
    parser.add_argument('--kilt_psgs', type=str)
    parser.add_argument('--train_aida_out', type=str)
    parser.add_argument('--kilt_psg_corpus', type=str)
    parser.add_argument('--train_trex_small', type=str)
    parser.add_argument('--train_zsre_small', type=str)
    parser.add_argument('--train_hopo_file', type=str)
    parser.add_argument('--train_hopo_query', type=str)
    parser.add_argument('--train_hopo_qrel', type=str)
    parser.add_argument('--dev_hopo_file', type=str)
    parser.add_argument('--dev_hopo_query', type=str)
    parser.add_argument('--dev_hopo_qrel', type=str)
    parser.add_argument('--train_nq_file', type=str)
    parser.add_argument('--train_nq_query', type=str)
    parser.add_argument('--train_nq_qrel', type=str)
    parser.add_argument('--dev_nq_file', type=str)
    parser.add_argument('--dev_nq_query', type=str)
    parser.add_argument('--dev_nq_qrel', type=str)
    parser.add_argument('--train_tqa_file', type=str)
    parser.add_argument('--train_tqa_query', type=str)
    parser.add_argument('--train_tqa_qrel', type=str)
    parser.add_argument('--dev_tqa_file', type=str)
    parser.add_argument('--dev_tqa_query', type=str)
    parser.add_argument('--dev_tqa_qrel', type=str)
    parser.add_argument('--process_nq', action='store_true')
    parser.add_argument('--process_tqa', action='store_true')
    parser.add_argument('--process_hopo', action='store_true')
    parser.add_argument('--process_wow', action='store_true')
    parser.add_argument('--process_trex', action='store_true')
    parser.add_argument('--process_fever', action='store_true')
    parser.add_argument('--process_zsre', action='store_true')
    parser.add_argument('--process_aida', action='store_true')
    parser.add_argument('--process_psgs', action='store_true')
    parser.add_argument('--process_docs', action='store_true')
    args = parser.parse_args()
    main(args)
