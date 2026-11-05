from pyfaidx import Fasta
import torch
# from random import random 
import random 
from pathlib import Path

import numpy as np

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
import torch.nn.functional as F

from pathlib import Path
# from pathlib import Path
import csv  # 添加支持 csv

import pandas as pd

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



def sliding_window(sequence, window_size, step_size):
    """
    从长序列中生成固定长度的滑动窗口子序列
    
    Args:
        sequence (str | list | torch.Tensor | np.ndarray): 输入序列
        window_size (int): 每个窗口的长度
        step_size (int): 滑动的步长

    Returns:
        List[sequence_type]: 一个窗口列表，每个元素是子序列
    """
    windows = []
    n = len(sequence)
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        windows.append(sequence[start:end])
    return windows


# ===== 测试 =====
# seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# windows = sliding_window(seq, window_size=5, step_size=3)

# for i, w in enumerate(windows):
#     print(f"Window {i}: {w}")


class NucleotideTransformerDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

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
        
        
        file = "tfsite"
        

        
        
        if file==1 :


            for file in (base_path.iterdir()):
                # if str(file).endswith('.fasta') and split in str(file):
                if str(file).endswith((".fasta", ".fa", ".fna")) and split in str(file):
                    self.seqs = Fasta(str(file), read_long_names=True )    

            self.label_mapper = {}
            for i, key in enumerate(self.seqs.keys()):
                self.label_mapper[i] = (key, int(key.rstrip()[-1]))
                
       

        elif file == "tfsite":
            # 数据集文件所在的根目录（所有 CSV 文件平铺在此目录下）
            tfsite_base_path = Path("./")
            # tfsite_base_path = Path("./")
            # 训练和测试文件路径
            train_file = tfsite_base_path / f"{dataset_name}_train.csv"
            test_file = tfsite_base_path / f"{dataset_name}_test.csv"
            print(f"[INFO] 训练数据文件: {train_file}")
            print(f"[INFO] 测试数据文件: {test_file}")
        
            all_train_data = []
            all_test_data = []
        
            # -------------------- 读取训练数据 --------------------
            if train_file.exists():
                try:
                    train_df = pd.read_csv(train_file)
                    if 'sequence' in train_df.columns and 'label' in train_df.columns:
                        for _, row in train_df.iterrows():
                            sequence = str(row['sequence']).strip()
                            label = int(row['label'])
                            all_train_data.append((sequence, label))
                        print(f"[INFO] 从 {train_file} 读取 {len(train_df)} 个训练样本")
                    else:
                        print(f"[WARNING] {train_file} 缺少 sequence 或 label 列")
                except Exception as e:
                    print(f"[ERROR] 读取 {train_file} 失败: {e}")
            else:
                print(f"[ERROR] 训练文件不存在: {train_file}")
        
            # -------------------- 读取测试数据 --------------------
            if test_file.exists():
                try:
                    test_df = pd.read_csv(test_file)
                    if 'sequence' in test_df.columns and 'label' in test_df.columns:
                        for _, row in test_df.iterrows():
                            sequence = str(row['sequence']).strip()
                            label = int(row['label'])
                            all_test_data.append((sequence, label))
                        print(f"[INFO] 从 {test_file} 读取 {len(test_df)} 个测试样本")
                    else:
                        print(f"[WARNING] {test_file} 缺少 sequence 或 label 列")
                except Exception as e:
                    print(f"[ERROR] 读取 {test_file} 失败: {e}")
            else:
                print(f"[ERROR] 测试文件不存在: {test_file}")
        
            print(f"[INFO] 总共读取 {len(all_train_data)} 个训练样本")
            print(f"[INFO] 总共读取 {len(all_test_data)} 个测试样本")
        
            # 按类别分组，保持验证集平衡（与原始逻辑一致）
            if all_train_data:
                random.seed(42)
        
                # 按标签分组
                label_groups = {}
                for sequence, label in all_train_data:
                    label_groups.setdefault(label, []).append((sequence, label))
        
                min_count = min(len(data_list) for data_list in label_groups.values())
                print(f"[INFO] 最小类别样本数: {min_count}")
                print(f"[INFO] 原始训练集类别分布: { {label: len(data_list) for label, data_list in label_groups.items()} }")
        
                # 平衡采样
                balanced_train_data = []
                for label, data_list in label_groups.items():
                    sampled = random.sample(data_list, min_count) if len(data_list) > min_count else data_list
                    balanced_train_data.extend(sampled)
        
                # 重新分组并打乱
                label_groups_balanced = {}
                for sequence, label in balanced_train_data:
                    label_groups_balanced.setdefault(label, []).append((sequence, label))
        
                for label in label_groups_balanced:
                    random.shuffle(label_groups_balanced[label])
        
                # 80/20 分割
                train_subset = []
                val_subset = []
                for label, data_list in label_groups_balanced.items():
                    split_idx = int(0.8 * len(data_list))
                    train_subset.extend(data_list[:split_idx])
                    val_subset.extend(data_list[split_idx:])
        
                random.shuffle(train_subset)
                random.shuffle(val_subset)
        
                print(f"[INFO] 平衡后训练集总数: {len(balanced_train_data)}")
                print(f"[INFO] 平衡后训练集类别分布: { {label: len(data_list) for label, data_list in label_groups_balanced.items()} }")
            else:
                train_subset, val_subset = [], []
                print("[WARNING] 没有读取到训练数据")
        
            # 根据 split 选择数据集
            if split == "train":
                self.seqs = all_train_data
            elif split == "val":
                self.seqs = all_test_data
            elif split == "test":
                self.seqs = all_test_data
            else:
                raise ValueError(f"Unknown split: {split}")
        
            self.label_mapper = {
                i: (sequence, label)
                for i, (sequence, label) in enumerate(self.seqs)
            }
        
            if self.seqs:
                label_counts = {}
                for _, label in self.seqs:
                    label_counts[label] = label_counts.get(label, 0) + 1
                print(f"[INFO] Loaded {len(self.seqs)} samples for split='{split}'")
                print(f"[INFO] Label distribution: {label_counts}")
            else:
                print(f"[INFO] Loaded {len(self.seqs)} samples for split='{split}'")




        else :
            self.seqs = []  # ⬅️ 移到外部

            for file in base_path.iterdir():
                if str(file).endswith(('.txt',)):  # 你只用 txt 就行
                    with open(file, 'r') as f:
                        seq_id, seq_lines = None, []
                        for line in f:
                            line = line.strip()
                            if line.startswith('>'):
                                if seq_id is not None:
                                    sequence = ''.join(seq_lines)
                                    self.seqs.append((seq_id, sequence))
                                seq_id = line[1:]
                                seq_lines = []
                            else:
                                seq_lines.append(line)
                        # 别忘最后一个
                        if seq_id is not None:
                            sequence = ''.join(seq_lines)
                            self.seqs.append((seq_id, sequence))

            # 标签提取
            self.label_mapper = {}
            for i, (seq_id, _) in enumerate(self.seqs):
                if "Positive" in seq_id:
                    label = 1
                elif "Negative" in seq_id:
                    label = 0
                else:
                    raise ValueError(f"Unknown label in: {seq_id}")
                self.label_mapper[i] = (seq_id, label)
                
                
                



    def __len__(self):
        # return len(self.seqs.keys())
        return len(self.seqs)


    def __getitem__(self, idx):
        # seq_id = self.label_mapper[idx][0]
        # if isinstance(self.seqs, list):
        # # 如果 self.seqs 是 list（即 file==0 的情况）
        # # label_mapper[i] = (seq_id, label) 中的顺序刚好和 seqs 一致
        # # 所以我们可以直接用 idx 来取
        #     _, x = self.seqs[idx]
        # else:
        #     # 如果 self.seqs 是字典（Fasta 对象，file==1 的情况）
        #     x = self.seqs[seq_id][:].seq
        # # x = self.seqs[seq_id][:].seq # only one sequence
        # y = self.label_mapper[idx][1] # 0 or 1 for binary classification
        # print(f"[DEBUG] idx={idx}, self.seqs[idx]={self.seqs[idx]}")

        
        
        seq_id = self.label_mapper[idx][0]
        # if isinstance(self.seqs, list):
        #     _, x = self.seqs[idx]
        # else:
        #     x = self.seqs[seq_id][:].seq
        
        if isinstance(self.seqs, list):
            x, _ = self.seqs[idx]  # 这里 sequence 是第一个元素
        else:
            x = self.seqs[seq_id][:].seq

        # print(f"Index {idx}: type(x)={type(x)}, x preview={str(x)[:30]}")

        y = self.label_mapper[idx][1]

        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)
            # print(f"After rc_aug: type(x)={type(x)}, x preview={str(x)[:30]}")

        # 确保是字符串
        assert isinstance(x, str), f"Error: sequence at idx {idx} is not string but {type(x)}"
        
        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)
        if self.tokenizer_name == 'dual':
           
            seq = x
            raw_seq = seq  # DNA字符串
            raw_seq1 = seq
            # char_tokenizer = self.tokenizer(0)
            # kmer_tokenizer = self.tokenizer(1)
            self.k = self.kmer_tokenizer.k  # 确保 k 被定义
            # Aux tokenizer: kmer
            # if len(seq) % self.k != 0:
            #     seq += 'N' * (self.k - len(seq) % self.k)
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
                raw_seq1,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=self.add_eos
            )
            seq2 = kmer_encoding["input_ids"]

            seq2 = torch.LongTensor(seq2)
            
#             expected_len = self.pad_max_length // self.k  # ensure consistent with tokenizer's max_length
#             if seq2.size(0) < expected_len:
#                 pad_id = getattr(self.kmer_tokenizer, 'pad_token_id', 0)
#                 seq2 = F.pad(seq2, (0, expected_len - seq2.size(0)), value=pad_id)

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
                max_length=self.max_length,
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
