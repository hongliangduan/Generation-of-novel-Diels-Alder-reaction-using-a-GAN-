import os
import random
import pickle
import pandas as pd

from copy import deepcopy

import torch
from tqdm.notebook import tqdm
from collections import defaultdict

data_path = '/home/tanshui/LS/SMILES-MaskGAN-main'
save_path = '/home/tanshui/LS/SMILES-MaskGAN-main'

encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R' }
decode_dict = {v: k for k, v in encode_dict.items()}


# Function definition for making data to be used for SMILES-MaskGAN
def get_pair(tokens, mask_idxs, mask_id):
    idxs = [vocab[atom] for atom in tokens]

    def _pad(ls, pad_index, max_length=100):
        padded_ls = deepcopy(ls)

        while len(padded_ls) <= max_length:
            padded_ls.append(pad_index)

        return padded_ls

    srcs = deepcopy(idxs)
    srcs.append(vocab['<eos>'])  # append eos id in srcs last

    tgts = deepcopy(idxs)
    tgts.insert(0, vocab['<eos>'])  # insert eos id in tgts first

    srcs_pad = _pad(srcs, vocab['<pad>'], max_length=100)
    tgts_pad = _pad(tgts, vocab['<pad>'], max_length=100)

    mask = torch.zeros(len(tgts_pad))
    for mask_idx in mask_idxs:
        offset = 1
        mask[mask_idx + offset] = 1
        srcs[mask_idx] = mask_id

    return srcs_pad, tgts_pad, len(srcs), mask


def encode(smiles: str) -> str:
    """
    Replace multi-char tokens with single tokens in SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        sanitized SMILE string with only single-char tokens
    """

    temp_smiles = smiles
    for symbol, token in encode_dict.items():
        temp_smiles = temp_smiles.replace(symbol, token)
    return temp_smiles


class Mask:
    mask_token = '__<m>__'

    def __call__(self, n):
        idxs = self.forward(n)

        # Verify indices are okay.
        assert (len(idxs) < n)

        valid_set = set(list(range(n)))
        for i in idxs:
            assert (i in valid_set)

        return idxs


class StochasticMask(Mask):
    def __init__(self, probability):
        self.p = probability
        self.r = random.Random(42)

    def forward(self, n):
        # Starting from one, since masks are messed,
        k = int(n * self.p)
        idxs = self.r.sample(range(1, n), k)

        return idxs


class VocabBuilder:
    def __init__(self, mask_builder):
        self.vocab_path = os.path.join(data_path, 'vocab', 'vocab.pt')

        self.mask_builder = mask_builder
        self._vocab = None

    def vocab(self):
        if self._vocab is None:
            self.build_vocab()

        return self._vocab

    def build_vocab(self):
        if os.path.exists(self.vocab_path):
            self._vocab = torch.load(self.vocab_path)

        else:
            self.rebuild_vocab()

    def rebuild_vocab(self):
        self.forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                  'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                  'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}

        self._vocab = {'<unk>': 0, '<pad>': 1, '<eos>': 2, '#': 20, '%': 22, '(': 25, ')': 24, '+': 26, '-': 27,
                       '.': 30,
                       '0': 32, '1': 31, '2': 34, '3': 33, '4': 36, '5': 35, '6': 38, '7': 37, '8': 40,
                       '9': 39, '=': 41, 'A': 7, 'B': 11, 'C': 19, 'F': 4, 'H': 6, 'I': 5, 'N': 10,
                       'O': 9, 'P': 12, 'S': 13, 'X': 15, 'Y': 14, 'Z': 3, '[': 16, ']': 18,
                       'b': 21, 'c': 8, 'n': 17, 'o': 29, 'p': 23, 's': 28, '>' : 46, '*' : 47,
                       "@": 42, "R": 43, '/': 44 , '\\' : 45 , 'l' : 48 , '__<m>__': 49
                       }

        self.idx_char = {v: k for k, v in self._vocab.items()}

        if self.vocab_path is not None:
            torch.save(self._vocab, self.vocab_path)


dir_ = os.path.join(data_path, 'data.csv')  # change data name
data = pd.read_csv(dir_)
smiles_list = [line.strip() for line in data['canonical_smiles']]
encoded_list = [encode(line) for line in smiles_list if len(line) <= 100]

rmask = StochasticMask(0.1)

vocab = None
if vocab is None:
    builder = VocabBuilder(rmask)
    vocab = builder.vocab()

final_data = defaultdict(
    list, {
        k: [] for k in ('train_srcs', 'train_tgts', 'train_lengths', 'train_mask')})

with tqdm(total=len(encoded_list)) as tbar:
    for i, tokens in enumerate(encoded_list):
        seq_len = len(tokens)
        mask_idxs = rmask(seq_len)
        mask_id = vocab['__<m>__']

        src, tgt, length, mask = get_pair(tokens, mask_idxs, mask_id)
        final_data['train_srcs'].append(src)
        final_data['train_tgts'].append(tgt)
        final_data['train_lengths'].append(length)
        final_data['train_mask'].append(mask)

        tbar.update(1)

# Save pickle file
with open(os.path.join(data_path, 'chembl26_canon_train_0.1.pkl'), 'wb') as f:
    pickle.dump(final_data, f, pickle.HIGHEST_PROTOCOL)

# load pickle file
with open(os.path.join(data_path, 'chembl26_canon_train_0.1.pkl'), 'rb') as f:
    val_data = pickle.load(f)

