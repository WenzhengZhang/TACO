import torch
from transformers import T5Tokenizer

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
a_tokenized = tokenizer(a,max_length=10,padding='max_length',truncation=True)
print(a_tokenized)