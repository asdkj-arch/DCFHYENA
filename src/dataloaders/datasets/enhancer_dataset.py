from pyfaidx import Fasta
import torch
from random import random 
from pathlib import Path

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
import torch.nn.functional as F

def coin_flip():
    return random() > 0.5

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


class EnhancerDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        char_tokenizer=None, 
        kmer_tokenizer=None,
        bpe_tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        pad_max_length=None,
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

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"

        # use Path object
        base_path = Path(dest_path) / dataset_name 
        assert base_path.exists(), 'path to file must exist'

#         for file in (base_path.iterdir()):
#             # if str(file).endswith('.fasta') and split in str(file):
#             if str(file).endswith((".fasta", ".fa", ".fna")) and split in str(file):
#                 self.seqs = Fasta(str(file), read_long_names=True )    

#         self.label_mapper = {}
#         for i, key in enumerate(self.seqs.keys()):
#             self.label_mapper[i] = (key, int(key.rstrip()[-1]))
            
            
            
        self.data = []  # ⬅️ 移到外部

        for file in base_path.iterdir():
            if str(file).endswith(('.txt',)):  # 你只用 txt 就行
                with open(file, 'r') as f:
                    seq_id, seq_lines = None, []
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if seq_id is not None:
                                sequence = ''.join(seq_lines)
                                self.data.append((seq_id, sequence))
                            seq_id = line[1:]
                            seq_lines = []
                        else:
                            seq_lines.append(line)
                    # 别忘最后一个
                    if seq_id is not None:
                        sequence = ''.join(seq_lines)
                        self.data.append((seq_id, sequence))

        # 标签提取
        self.label_mapper = {}
        for i, (seq_id, _) in enumerate(self.data):
            if "Positive" in seq_id:
                label = 1
            elif "Negative" in seq_id:
                label = 0
            else:
                raise ValueError(f"Unknown label in: {seq_id}")
            self.label_mapper[i] = (seq_id, label)



    def __len__(self):
        return len(self.seqs.keys())

    def __getitem__(self, idx):
        seq_id = self.label_mapper[idx][0]
        x = self.seqs[seq_id][:].seq # only one sequence
        y = self.label_mapper[idx][1] # 0 or 1 for binary classification
        
        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)
        if self.tokenizer_name == 'dual':
           
            seq = x
            raw_seq = seq  # DNA字符串
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
            seq1 = char_seq["input_ids"]
            seq1 = torch.LongTensor(seq1)



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
            target1 = torch.LongTensor([y])  # offset by 1, includes eos
            data2 = seq2  # remove eos
            target2 = torch.LongTensor([y])  # offset by 1, includes eos

            return data1 , target1 , data2 , target2
        
        if self.tokenizer_name == 'dual1':
           
            seq = x
            raw_seq = seq  # DNA字符串
            # char_tokenizer = self.tokenizer(0)
            # kmer_tokenizer = self.tokenizer(1)
            

            char_seq = self.char_tokenizer(
                seq,
                add_special_tokens=True if self.add_eos else False,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq1 = char_seq["input_ids"]
            seq1 = torch.LongTensor(seq1)



            bpe_encoding = self.bpe_tokenizer(
                raw_seq,
                padding="max_length",
                truncation=True,
                add_special_tokens=self.add_eos
            )
            seq2 = bpe_encoding["input_ids"]
            seq2 = torch.LongTensor(seq2)
            # seq2 = F.pad(seq2, (0, self.max_length - 1 - seq2.size(0)), value=self.bpe_tokenizer.pad_token_id)
            seq2 = F.pad(seq2, (0, self.max_length - seq2.size(0)), value=self.bpe_tokenizer.pad_token_id)

            data1 = seq1  # remove eos
            target1 = torch.LongTensor([y])  # offset by 1, includes eos
            data2 = seq2  # remove eos
            target2 = torch.LongTensor([y])  # offset by 1, includes eos

            return data1 , target1 , data2 , target2
        
        
        elif self.tokenizer_name == 'dual2':
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
            seq1 = F.pad(seq1, (0, self.max_length - 1 - seq1.size(0)), value=self.bpe_tokenizer.pad_token_id)



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
            target1 = torch.LongTensor([y]) # offset by 1, includes eos
            data2 = seq2  # remove eos
            target2 = torch.LongTensor([y])  # offset by 1, includes eos

            return data1 , target1 , data2 , target2


        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        seq_ids = torch.LongTensor(seq_ids)

        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        # need to wrap in list
        target = torch.LongTensor([y])  # offset by 1, includes eos

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
