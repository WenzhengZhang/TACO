import torch
from transformers import T5Tokenizer, BertTokenizer
from collections import defaultdict
from copy import deepcopy
import itertools
from torch.optim import adamw

# a = torch.randn((3, 4)).bool()
# print(a)
# b = torch.block_diag(*a)
# print(b)
# print(b.size())
# c = torch.zeros((3, 12))
# for i in range(a.size(0)):
#     c[i, 4 * i:4 * i + 4] = a[i]
# print(b == c)
a = {'1': 1, '2': 2}
b = list(a.keys())
print(b)
b.remove('2')
print(b)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
a = 'how are you doing'
a_enc = tokenizer.encode(a, add_special_tokens=False, max_length=10,
                         truncation=True)
print(a_enc)
a_enc_plus = tokenizer.encode_plus(a_enc,
                                   truncation='only_first',
                                   max_length=10,
                                   padding=False,
                                   return_attention_mask=False,
                                   return_token_type_ids=False,
                                   )

print(a_enc_plus)
a = torch.randn((6, 4))
b = a.view((2, 3, 4))
print(b.size())
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
if isinstance(a[0], list):
    b = sum(a, [])
    print(b)
a = 'how is it going'
a_tokenized = tokenizer(a, max_length=10, padding='max_length', truncation=True)
print(a_tokenized)
a = 'how,'
print(a.split(','))


def shuffle_cycle(iterable):
    while True:
        for x in iterable:
            yield x


a = [[1, 2], [7, 8]]
b = [[3, 4], [10, 11], [12, 13]]
c = [[6, 7]]
loaders = [a, b, c]
d = iter(zip(*(shuffle_cycle(loader) if len(loader) < 3 else loader for
               loader in
               loaders)))
for k in d:
    print(k)
s = " "
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('t5 tokenize')
print(tokenizer.tokenize(s))
print('bert tokenize')
print(bert_tokenizer.tokenize(s))
a = defaultdict(list)
a[1].append(3)
a[2].append(4)
a[1].append(5)
print(a[1][-1] == 5)
print(5 in a[1])
a = torch.tensor([float('nan'), float('inf'), 1.0])
print(torch.isfinite(a))
b = torch.isfinite(a).all()
print(b)
if b:
    print('hey')
else:
    print('ha')
print(a.view(-1))

a = torch.tensor([[0.1, 1, 2, 3], [0.2, 4, 5, 6]])
b = torch.tensor([0.1, 0.5, 4, 1, 6, 3])
c = torch.isin(a, b, invert=True)
print(c)
d = a.masked_fill(c, float('-inf'))
e = d.amax(-1)
print(e)
a = torch.randn((3, 4))
b = torch.tensor([2, 1, 0]).long()
a[torch.arange(3), b] = float('inf')
print(a)
m = torch.ones((5, 8), dtype=torch.bool)
s = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 0]],
                 dtype=torch.bool)
m[:, 4:7] = s
print(m)
print(a[torch.arange(3), 2])
print(a[:, 2])
is_eos = torch.tensor([0, 1, 1], dtype=torch.bool)
a[is_eos] = float('nan')
print(a)
c = torch.randint(10, (3, 6))
print(c)
d = torch.tensor([[1, 1, 5, 4], [2, 1, 3, 2], [0, 0, 0, 3]]).long()
c.scatter_(1, d, 100)
print(c)
# p = torch.randn((3, 6))
# p1 = deepcopy(p)
# k = torch.tensor([2, 1, 3]).long()
# p[torch.arange(3), k] = float('nan')
# p1.scatter_(1, k, float('nan'))
# print(p)
# print(p1)
e = torch.tensor([[2, 2, 1, 4], [3, 0, 1, 0]]).long()
c[1:3].scatter_(1, e, 1000)
print(c)
p = torch.ones((4, 6), dtype=torch.bool)
m = torch.tensor([False, True, False, True]).bool()
ind = torch.tensor(
    [[1, 1, 0, 2], [0, 1, 4, 0], [3, 3, 0, 5], [3, 3, 1, 4]]).long()
# p[m].scatter_(1, ind[m], False)
# print(p[m])
p[m] = p[m].scatter(1, ind[m], False)
print(p)
a = torch.randn((3, 4))
print(a)
a[a.amax(-1) > 0.1] += 1
print(a)
a = torch.randn((3, 4))
b = a[torch.arange(3), torch.tensor([0, 2, 1])]
print(a)
print(b)
print(a == b.unsqueeze(1))


def split_list(ls, delimiter, include_delimiter):
    if not include_delimiter:
        spl = [list(y) for x, y in itertools.groupby(
            ls, lambda z: z == delimiter) if
               not x]
    else:
        spl = []
        for x, y in itertools.groupby(ls, lambda z: z == delimiter):
            if x:
                spl.append([])
            spl[-1].extend(y)
    return spl


a = ['<s>', 'this', 'is', '</s>', '<s>', 'a', 'sentence', '</s>']
b = split_list(a, '<s>', True)
c = split_list(a, '<s>', False)
print(b)
print(c)
a= 2
b = int(2)
print(b)

a = torch.randn((3,8))
b = torch.randn((3,8))
c = (a.softmax(0)*b).sum(0)
print(c)
a_chunk = a.unfold(0,2,2)
b_chunk = b.unfold(0,2,2)
d = (a_chunk.softmax(0)*b_chunk).sum(0)
print(d)

