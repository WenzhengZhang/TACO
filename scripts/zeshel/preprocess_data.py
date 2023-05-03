import argparse
import json
import csv
import os


def load_data(data_dir):
    """
    :param data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')
    mention_path = os.path.join(data_dir, 'mentions')

    def load_mentions(part):
        mentions = []
        domains = set()
        with open(os.path.join(mention_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                mentions.append(field)
                domains.add(field['corpus'])
        return mentions, domains

    samples_train, train_domain = load_mentions('train')
    samples_val, val_domain = load_mentions('val')
    samples_test, test_domain = load_mentions('test')

    def load_entities(domains):
        """
        :param domains: list of domains
        :return: all the entities in the domains
        """
        doc = {}
        doc_path = os.path.join(data_dir, 'documents')
        for domain in domains:
            with open(os.path.join(doc_path, domain + '.json')) as f:
                for line in f:
                    field = json.loads(line)
                    page_id = field['document_id']
                    doc[page_id] = field
        return doc

    train_doc = load_entities(train_domain)
    val_doc = load_entities(val_domain)
    test_doc = load_entities(test_domain)

    return samples_train, samples_val, samples_test, \
           train_doc, val_doc, test_doc


def process_query(samples, doc, output_dir, split, max_len):
    query_path = os.path.join(output_dir, f"{split}.query.txt")
    if split == 'train':
        ext = 'tsv'
    else:
        ext = 'trec'
    qrel_path = os.path.join(output_dir, f"{split}.qrel.{ext}")
    with open(query_path, 'w') as f_query, open(qrel_path, 'w') as f_qrel:
        tsv_query = csv.writer(f_query, delimiter="\t")
        delimiter_qrel = "\t" if split == 'train' else " "
        tsv_qrel = csv.writer(f_qrel, delimiter=delimiter_qrel)
        for item in samples:
            qid = item["mention_id"]
            ctxt = doc[item["context_document_id"]]
            query_text = get_query_text(ctxt, item["start_index"],
                                        item["end_index"], max_len)
            d_id = item["label_document_id"]
            tsv_query.writerow([qid, query_text])
            tsv_qrel.writerow([qid, 0, d_id, 1])


def process_corpus(doc, output_dir, split):
    corpus_path = os.path.join(output_dir, f"psg_corpus_{split}.tsv")
    with open(corpus_path, "w") as f:
        tsv_w = csv.writer(f, delimiter="\t")
        for item in doc.values():
            doc_id = item["document_id"]
            title = item["title"]
            text = item["text"]
            tsv_w.writerow([doc_id, title, text])


def get_query_text(ctxt, start_index, end_index, max_len):
    tokens = ctxt["text"].split()
    ctx_l = tokens[max(0, start_index - max_len - 1):
                   start_index]
    ctx_r = tokens[end_index + 1:
                   end_index + max_len + 2]
    mention = tokens[start_index:end_index + 1]
    if len(mention) >= max_len:
        text = mention
    else:
        leftover = max_len - len(mention) - 2  # <m> token and </m> token
        leftover_hf = leftover // 2
        if len(ctx_l) > leftover_hf:
            ctx_l_len = leftover_hf if len(
                ctx_r) > leftover_hf else leftover - len(ctx_r)
        else:
            ctx_l_len = len(ctx_l)
        text = ctx_l[-ctx_l_len:] + ["<m>"] + mention + ["</m>"] + ctx_r
    text = " ".join(text[:max_len])
    return text


def main(args):
    print("load original data ... ")
    samples_train, samples_val, samples_test, \
    train_doc, val_doc, test_doc = load_data(args.input_dir)
    print('process train corpus ... ')
    process_corpus(train_doc, args.output_dir, 'train')
    print('process dev corpus ... ')
    process_corpus(val_doc, args.output_dir, 'dev')
    print('process test corpus ... ')
    process_corpus(test_doc, args.output_dir, 'test')
    print('process train query and qrel ... ')
    process_query(samples_train, train_doc, args.output_dir, 'train',
                  args.max_len)
    print('process dev query and qrel ... ')
    process_query(samples_val, val_doc, args.output_dir, 'dev', args.max_len)
    print('process test query and qrel')
    process_query(samples_test, test_doc, args.output_dir, 'test', args.max_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=180,
                        help="max query text length")
    args = parser.parse_args()
    main(args)
