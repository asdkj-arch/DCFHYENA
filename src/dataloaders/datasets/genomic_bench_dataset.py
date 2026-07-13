from itertools import islice
from functools import partial
import os
import functools
# import json
# from pathlib import Path
# from pyfaidx import Fasta
# import polars as pl
# import pandas as pd
import torch
from random import randrange, random
import numpy as np
from pathlib import Path
# import random



from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded
from src.dataloaders.base import default_data_path
import torch.nn.functional as F


# import random
import math
# from collections import Counter

"""

Genomic Benchmarks Dataset, from:
https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks


"""


# helper functions

def exists(val):
    return val is not None

def identity(t):
    return t

def cast_list(t):
    return t if isinstance(t, list) else [t]

def coin_flip():
    return random() > 0.5

# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord('a')] = 0
seq_indices_embed[ord('c')] = 1
seq_indices_embed[ord('g')] = 2
seq_indices_embed[ord('t')] = 3
seq_indices_embed[ord('n')] = 4
seq_indices_embed[ord('A')] = 0
seq_indices_embed[ord('C')] = 1
seq_indices_embed[ord('G')] = 2
seq_indices_embed[ord('T')] = 3
seq_indices_embed[ord('N')] = 4
seq_indices_embed[ord('.')] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype = np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims = (-1,))

def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, 'must be one hot encoding with last dimension equal to 4'
    return torch.flip(one_hot, (-1, -2))



class GenomicBenchmarkDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name="human_nontata_promoters",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        char_tokenizer=None, 
        kmer_tokenizer=None,
        bpe_tokenizer=None,
        tokenizer_name=None,
        #use_padding=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        pad_max_length=None,
        # val_fraction=0.1,   # 新增：train 切分 val 的比例
        # seed=42             # 新增：随机种子，保证可复现
    ):

        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.char_tokenizer = char_tokenizer
        self.kmer_tokenizer = kmer_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print("downloading {} to {}".format(dataset_name, dest_path))
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print("already downloaded {}-{}".format(split, dataset_name))

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"
        
#                 # 1) 检查数据是否已下载
#         if not is_downloaded(dataset_name, cache_path=dest_path):
#             print(f"Downloading {dataset_name} to {dest_path}")
#             download_dataset(dataset_name, version=0, dest_path=dest_path)
#         else:
#             print(f"Already downloaded {dataset_name} - split: {split}")

#         base_path = Path(dest_path) / dataset_name
        
        
#         if split == "val":
#             if (base_path / "val").exists():
#                 base_path = base_path / "val"
#             else:
#                 print("[Info] No explicit val split found. Splitting train data.")
#                 all_train = list((base_path / "train").glob("*/*.txt"))
#                 random.seed(seed)
#                 random.shuffle(all_train)
#                 val_size = int(len(all_train) * val_fraction)
#                 self.file_list = all_train[:val_size]
#         elif split in ["train", "test"]:
#             base_path = base_path / split
#             self.file_list = list(base_path.glob("*/*.txt"))
#         else:
#             raise ValueError(f"Unknown split: {split}")

        # use Path object
        base_path = Path(dest_path) / dataset_name / split

        self.all_seqs = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for path in (base_path / label_type).iterdir():
                with open(path, "r") as f:
                    content = f.read()
                self.all_seqs.append(content)
                self.all_labels.append(label_mapper[label_type])
                
    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        
        if self.tokenizer_name == 'dual':
           
            x = self.all_seqs[idx]
            y = self.all_labels[idx]
            if self.rc_aug and coin_flip():
                x1 = string_reverse_complement(x)
            # x1 = string_reverse_complement(x)
            seq = x
            raw_seq = seq  # DNA字符串
            # raw_seq1 = x1
            # char_tokenizer = self.tokenizer(0)
            # kmer_tokenizer = self.tokenizer(1)
            self.k = self.kmer_tokenizer.k  # 确保 k 被定义
            # Aux tokenizer: kmer
            if len(seq) % self.k != 0:
                seq += 'N' * (self.k - len(seq) % self.k)
            char_seq = self.char_tokenizer(
                seq,
                add_special_tokens=True if self.add_eos else False,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            # seq1 = char_seq["input_ids"]
            # seq1 = torch.LongTensor(seq1).clone()
            
            seq1 = torch.tensor(char_seq["input_ids"], dtype=torch.long).clone().contiguous()
            
            # char_seq1 = self.char_tokenizer(
            #     raw_seq1,
            #     add_special_tokens=True if self.add_eos else False,
            #     padding="max_length",
            #     max_length=self.max_length,
            #     truncation=True,
            # )
            # seq2 = char_seq["input_ids"]
            # seq2 = torch.LongTensor(seq2)

            k = 3
            if len(seq) % k != 0:
                seq += 'N' * (k - len(seq) % k)
            kmer_max_len = math.ceil(self.max_length / k)

            kmer_encoding = self.kmer_tokenizer(
                raw_seq,
                padding="max_length",
                # max_length=self.max_length,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=self.add_eos
            )
#             seq2 = kmer_encoding["input_ids"]

#             seq2 = torch.LongTensor(seq2).clone()
            seq2 = torch.tensor(kmer_encoding["input_ids"], dtype=torch.long).clone().contiguous()
            
            # expected_len = self.pad_max_length // self.k  # ensure consistent with tokenizer's max_length
            # if seq2.size(0) < expected_len:
            #     pad_id = getattr(self.kmer_tokenizer, 'pad_token_id', 0)
            #     seq2 = F.pad(seq2, (0, expected_len - seq2.size(0)), value=pad_id)
            
            
            # print(f"[Debug] idx={idx}, seq1.shape={seq1.shape}, seq2.shape={seq2.shape}")
            



            data1 = seq1  # remove eos
            target1 = self.all_labels[idx]  # offset by 1, includes eos
            data2 = seq2  # remove eos
            target2 = self.all_labels[idx]  # offset by 1, includes eos

            return data1 , target1 , data2 , target2
        
        if self.tokenizer_name == 'dual1':
           
            x = self.all_seqs[idx]
            y = self.all_labels[idx]
            # if self.rc_aug and coin_flip():
            #     x = string_reverse_complement(x)
                
            x1 = string_reverse_complement(x)
            seq = x
            raw_seq = seq  # DNA字符串
            raw_seq1 = x1
            # char_tokenizer = self.tokenizer(0)
            # kmer_tokenizer = self.tokenizer(1)
            

            char_seq = self.char_tokenizer(
                seq,
                add_special_tokens=True if self.add_eos else False,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            
            # char_seq = self.bpe_tokenizer(
            #     seq,
            #     padding="max_length",
            #     max_length=self.max_length,
            #     truncation=True,
            #     add_special_tokens=self.add_eos
            # )
            seq1 = char_seq["input_ids"]
            seq1 = torch.LongTensor(seq1)



            bpe_encoding = self.bpe_tokenizer(
                raw_seq1,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=self.add_eos
            )
            seq2 = bpe_encoding["input_ids"]
            seq2 = torch.LongTensor(seq2)
            # seq2 = F.pad(seq2, (0, self.max_length - 1 - seq2.size(0)), value=self.bpe_tokenizer.pad_token_id)
            seq2 = F.pad(seq2, (0, self.max_length - seq2.size(0)), value=self.bpe_tokenizer.pad_token_id)


            data1 = seq1  # remove eos
            target1 = self.all_labels[idx]  # offset by 1, includes eos
            data2 = seq2  # remove eos
            target2 = self.all_labels[idx]  # offset by 1, includes eos

            return data1 , target1 , data2 , target2
        
        
        elif self.tokenizer_name == 'dual2':
            x = self.all_seqs[idx]
            y = self.all_labels[idx]
            seq = x
            raw_seq = seq  # DNA字符串
            # bpe1_tokenizer = self.tokenizer(0)
            # kmer1_tokenizer = self.tokenizer(1)
            # Aux tokenizer: kmer
            self.k = self.kmer_tokenizer.k  # 确保 k 被定义
            bpe_encoding = self.bpe_tokenizer(
                raw_seq,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq1 = bpe_encoding["input_ids"]    
            seq1 = torch.LongTensor(seq1)
            # seq1 = F.pad(seq1, (0, self.max_length - 1 - seq1.size(0)), value=self.bpe_tokenizer.pad_token_id)
            seq1 = F.pad(seq1, (0, self.max_length - seq1.size(0)), value=self.bpe_tokenizer.pad_token_id)



            kmer_encoding = self.kmer_tokenizer(
                raw_seq,
                padding="max_length",
                max_length=self.pad_max_length // self.k,
                truncation=True,
                add_special_tokens=self.add_eos
            )
            seq2 = kmer_encoding["input_ids"]
            seq2 = torch.LongTensor(seq2)
            expected_len = self.pad_max_length // self.k  # ensure consistent with tokenizer's max_length
            if seq2.size(0) < expected_len:
                pad_id = getattr(self.kmer_tokenizer, 'pad_token_id', 0)
                seq2 = F.pad(seq2, (0, expected_len - seq2.size(0)), value=pad_id)

            data1 = seq1  # remove eos
            target1 = self.all_labels[idx]  # offset by 1, includes eos
            data2 = seq2  # remove eos
            target2 = self.all_labels[idx]  # offset by 1, includes eos

            return data1 , target1 , data2 , target2
        x = self.all_seqs[idx]
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
           # padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_ids)

        # need to wrap in list
        target = torch.LongTensor([y])  # offset by 1, includes eos

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target






if __name__ == '__main__':
    """Quick test loading dataset.
    
    example
    python -m src.dataloaders.datasets.genomic_bench_dataset
    
    """

    max_length = 300  # max len of seq grabbed
    use_padding = True
    dest_path = "data/genomic_benchmark/"
    return_mask = True
    add_eos = True
    padding_side = 'right'    

    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length,
        add_special_tokens=False,
        padding_side=padding_side,
    )

    ds = GenomicBenchmarkDataset(
        max_length = max_length,
        use_padding = use_padding,
        split = 'train', # 
        tokenizer=tokenizer,
        tokenizer_name='char',
        dest_path=dest_path,
        return_mask=return_mask,
        add_eos=add_eos,
    )

    # it = iter(ds)
    # elem = next(it)
    # print('elem[0].shape', elem[0].shape)
    # print(elem)
    # breakpoint()
