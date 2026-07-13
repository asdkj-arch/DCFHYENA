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
        
        
        file = "tfsite2"
        
        #     # change "val" split to "test".  No val available, just test
        if file=="tfsite2" :
            if split == "val":
                split = "test"
                
        # if file==4 :
        #     if split == "val":
        #         split = "dev"
                

            # use Path object
        # base_path = Path(dest_path) / dataset_name 
        # assert base_path.exists(), 'path to fasta file must exist'
        
        
        
        if file==1 :


            for file in (base_path.iterdir()):
                # if str(file).endswith('.fasta') and split in str(file):
                if str(file).endswith((".fasta", ".fa", ".fna")) and split in str(file):
                    self.seqs = Fasta(str(file), read_long_names=True )    

            self.label_mapper = {}
            for i, key in enumerate(self.seqs.keys()):
                self.label_mapper[i] = (key, int(key.rstrip()[-1]))
                
        elif file == 2:


            self.seqs = []
            for file in base_path.iterdir():
                if str(file).endswith(".csv") and split in str(file):
                    with open(file, "r") as f:
                        reader = csv.DictReader(f)
                        # reader = csv.DictReader(f, delimiter="\t") 
                        for row in reader:
                                # 只保存 sequence 和 label，不保存 chr_id
                            sequence = row["sequence"]
                            label = int(row["label"])
                            self.seqs.append((sequence, label))
                            
                            # print(self.seqs[0])
                            # print(type(self.seqs[0][0]))


                # 构造 label_mapper，索引到 (sequence, label)
            self.label_mapper = {i: (sequence, label) for i, (sequence, label) in enumerate(self.seqs)}
            
            
            
        elif file == 3:
            self.seqs = []
            for txt_file in base_path.iterdir():
                if str(txt_file).endswith(".txt") and split in str(txt_file):
                    with open(txt_file, "r") as f:
                        current_header = None
                        current_sequence = []

                        for line in f:
                            line = line.strip()
                            # 检测序列头（以 > 开头）
                            if line.startswith(">"):
                                # 保存前一个序列（如果有）
                                if current_header is not None:
                                    full_sequence = ''.join(current_sequence)
                                    # 确定标签：P->1 (正样本), N->0 (负样本)
                                    label = 1 if current_header.startswith("P") else 0
                                    self.seqs.append((full_sequence, label))

                                # 开始新序列
                                current_header = line[1:].split()[0]  # 去掉 > 并取第一个字段
                                current_sequence = []
                            elif line:  # 非空序列行
                                current_sequence.append(line)

                        # 处理最后一个序列
                        if current_header is not None and current_sequence:
                            full_sequence = ''.join(current_sequence)
                            label = 1 if current_header.startswith("P") else 0
                            self.seqs.append((full_sequence, label))

                # 构造 label_mapper
                self.label_mapper = {i: (seq, label) for i, (seq, label) in enumerate(self.seqs)}
                
                
        elif file == 4:

            self.seqs = []
            for file in base_path.iterdir():
                if str(file).endswith(".csv") and split in str(file):
                    with open(file, "r") as f:
                        reader = csv.DictReader(f)  # 自动用第一行的表头作为 key
                        for row in reader:
                            sequence = row["sequence"].strip()
                            label = int(row["label"])
                            self.seqs.append((sequence, label))

            # 构造 label_mapper，索引到 (sequence, label)
            self.label_mapper = {i: (sequence, label) for i, (sequence, label) in enumerate(self.seqs)}
            
            
        elif file == 5:
            self.seqs = []

            # 确定要读取的目录
            data_dir = base_path / split if split else base_path

            # 检查目录是否存在
            if not data_dir.exists() or not data_dir.is_dir():
                raise ValueError(f"Directory {data_dir} does not exist or is not a directory")

            # 查找positive和negative文件
            positive_file = None
            negative_file = None
            for item in data_dir.iterdir():
                if item.is_file():
                    if item.name == "positive":
                        positive_file = item
                    elif item.name == "negative":
                        negative_file = item

            # 如果找不到文件，尝试在子目录中查找
            if positive_file is None or negative_file is None:
                for subdir in data_dir.iterdir():
                    if subdir.is_dir():
                        for item in subdir.iterdir():
                            if item.is_file():
                                if item.name == "positive":
                                    positive_file = item
                                elif item.name == "negative":
                                    negative_file = item

            # 检查是否找到了文件
            if positive_file is None:
                raise FileNotFoundError(f"Positive file not found in {data_dir}")
            if negative_file is None:
                raise FileNotFoundError(f"Negative file not found in {data_dir}")

            # 读取positive文件
            with open(positive_file, "r") as f:
                current_header = None
                current_sequence = []

                for line in f:
                    line = line.strip()
                    # 检测序列头（以 > 开头）
                    if line.startswith(">"):
                        # 保存前一个序列（如果有）
                        if current_header is not None and current_sequence:
                            full_sequence = ''.join(current_sequence)
                            self.seqs.append((full_sequence, 1))  # 正样本标签为1

                        # 开始新序列
                        current_header = line[1:]  # 去掉 > 
                        current_sequence = []
                    elif line:  # 非空序列行
                        current_sequence.append(line)

                # 处理最后一个序列
                if current_header is not None and current_sequence:
                    full_sequence = ''.join(current_sequence)
                    self.seqs.append((full_sequence, 1))

            # 读取negative文件
            with open(negative_file, "r") as f:
                current_header = None
                current_sequence = []

                for line in f:
                    line = line.strip()
                    # 检测序列头（以 > 开头）
                    if line.startswith(">"):
                        # 保存前一个序列（如果有）
                        if current_header is not None and current_sequence:
                            full_sequence = ''.join(current_sequence)
                            self.seqs.append((full_sequence, 0))  # 负样本标签为0

                        # 开始新序列
                        current_header = line[1:]  # 去掉 > 
                        current_sequence = []
                    elif line:  # 非空序列行
                        current_sequence.append(line)

                # 处理最后一个序列
                if current_header is not None and current_sequence:
                    full_sequence = ''.join(current_sequence)
                    self.seqs.append((full_sequence, 0))

            # 构造 label_mapper
            self.label_mapper = {i: (seq, label) for i, (seq, label) in enumerate(self.seqs)}
            
            
            
            
            
        elif file == 6:
            # txt 格式 + 滑动窗口
            base_path = Path(dest_path) / dataset_name
            txt_file = None
            for file in base_path.iterdir():
                if str(file).endswith(".txt") and split in str(file):
                    txt_file = file
                    break
            assert txt_file is not None, f"No txt file found for split={split}"

            # 读取整条序列
            with open(txt_file, "r") as f:
                sequence = f.read().strip()

            # 切成滑动窗口 (默认窗口=65，步长=1)
            windows = sliding_window(sequence, window_size=65, step_size=1)

            # 标签全为 0
            self.seqs = [(win, 0) for win in windows]
            self.label_mapper = {i: (win, 0) for i, win in enumerate(windows)}
            
            
            
        elif file == 7:


            self.seqs = []

            # 配置参数
            rbp_name = "AUF1"
            window_size = 257
            base_path = Path("/data/wuhehe/wuhe/Bsite_data_200_6000_lower_3site")

            seq_file = base_path / rbp_name / f"{rbp_name}_seq.fasta"
            label_file = base_path / rbp_name / f"{rbp_name}_train.fasta"  # 也可以改成 test.fasta

            # --- 读取所有序列 ---
            all_seqs = {}
            with open(seq_file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith(">"):
                        name = line.strip()[1:]
                        all_seqs[name] = lines[i + 1].strip()

            # --- 读取标注 ---
            seq_labels = {}
            with open(label_file, "r") as f:
                lines = f.readlines()
                current_name = None
                for line in lines:
                    if line.startswith(">"):
                        current_name = line.strip()[1:]
                        seq_labels[current_name] = np.zeros(len(all_seqs[current_name]), dtype=int)
                    else:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            start, end = int(parts[-2]), int(parts[-1])
                            seq_labels[current_name][start - 1:end] = 1

            # --- 窗口化生成正负样本 ---
            for name, seq in all_seqs.items():
                y = seq_labels[name]
                for i in range(len(seq)):
                    if i < window_size // 2:
                        subseq = (window_size - len(seq[:i + window_size // 2]) - 1) * '0' + seq[:i + window_size // 2 + 1]
                    elif i >= len(seq) - window_size // 2:
                        subseq = seq[i - window_size // 2:] + '0' * (window_size - len(seq[i - window_size // 2:]))
                    else:
                        subseq = seq[i - window_size // 2:i + window_size // 2 + 1]
                    label = int(y[i])
                    self.seqs.append((subseq, label))

            # 构造 label_mapper
            self.label_mapper = {i: (seq, label) for i, (seq, label) in enumerate(self.seqs)}


        elif file == "tfsite":

            # 设置数据集基础路径
            tfsite_base_path = Path("/public/ojsys/eye/wuwencan/lihaokai/dataset/hyena/tfbs_individual/backup/")
            base_path = tfsite_base_path / dataset_name
        
            print(f"[INFO] 数据集路径: {base_path}")
        
            all_train_data = []
            all_test_data = []
        
            # -------------------- 读取训练数据 --------------------
            train_dir = base_path / "train"
            if train_dir.exists():
                # 优先读取标准的 train.csv
                train_file = train_dir / "train.csv"
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
                    # 若 train.csv 不存在，则读取 train 目录下所有 CSV 文件
                    csv_files = list(train_dir.glob("*.csv"))
                    if csv_files:
                        for csv_file in csv_files:
                            try:
                                df = pd.read_csv(csv_file)
                                if 'sequence' in df.columns and 'label' in df.columns:
                                    for _, row in df.iterrows():
                                        sequence = str(row['sequence']).strip()
                                        label = int(row['label'])
                                        all_train_data.append((sequence, label))
                                    print(f"[INFO] 从 {csv_file} 读取 {len(df)} 个训练样本")
                            except Exception as e:
                                print(f"[ERROR] 读取 {csv_file} 失败: {e}")
                    else:
                        print(f"[WARNING] {train_dir} 下未找到任何 CSV 文件")
            else:
                print(f"[ERROR] 训练目录不存在: {train_dir}")
        
            # -------------------- 读取测试数据 --------------------
            test_dir = base_path / "test"
            if test_dir.exists():
                test_file = test_dir / "test.csv"
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
                    csv_files = list(test_dir.glob("*.csv"))
                    if csv_files:
                        for csv_file in csv_files:
                            try:
                                df = pd.read_csv(csv_file)
                                if 'sequence' in df.columns and 'label' in df.columns:
                                    for _, row in df.iterrows():
                                        sequence = str(row['sequence']).strip()
                                        label = int(row['label'])
                                        all_test_data.append((sequence, label))
                                    print(f"[INFO] 从 {csv_file} 读取 {len(df)} 个测试样本")
                            except Exception as e:
                                print(f"[ERROR] 读取 {csv_file} 失败: {e}")
                    else:
                        print(f"[WARNING] {test_dir} 下未找到任何 CSV 文件")
            else:
                print(f"[ERROR] 测试目录不存在: {test_dir}")
        
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
                self.seqs = train_subset
            elif split == "val":
                self.seqs = val_subset
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




        elif file == "tfsite2":
            # 数据集文件所在的根目录（所有 CSV 文件平铺在此目录下）
            tfsite_base_path = Path("/public/ojsys/eye/wuwencan/lihaokai/dataset/hyena/tfbs_individual/back_165/")
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



        elif file == "tfsite1":
            tfsite_base_path = Path("/public/ojsys/eye/wuwencan/lihaokai/dataset/hyena/tfbs_individual/backup/")
            # 确保 dataset_name 以 .csv 结尾
    
            base_dir = tfsite_base_path / dataset_name
            file_path = base_dir / "test" / "test.csv"
        
            self.seqs = []
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                # 检查必需的列是否存在
                if 'sequence' not in reader.fieldnames or 'label' not in reader.fieldnames:
                    raise ValueError(f"文件 {file_path} 缺少 'seq' 或 'label' 列，可用列: {reader.fieldnames}")
                
                for row in reader:
                    sequence = row['sequence'].strip()
                    label = row['label']
                    # 如果标签是字符串，尝试转换为整数；可根据实际数据调整类型转换
                    try:
                        label = int(label)
                    except ValueError:
                        # 如果无法转换为整数，保持原样（后续可能需要映射）
                        pass
                    if sequence:   # 忽略空序列
                        self.seqs.append((sequence, label))
            
            self.label_mapper = dict(enumerate(self.seqs))



        elif file == "tfbs_test":
            # base_dir = Path("/public/ojsys/eye/wuwencan/lihaokai/dataset/gwas/Glaucoma/tfbs_by_factor_cell_split_filtered")
            
            base_dir = Path("/public/ojsys/eye/wuwencan/lihaokai/dataset/gwas/Glaucoma/lunwen/")
            # 确保 dataset_name 以 .csv 结尾（如果可能不带后缀，则添加）
            if not dataset_name.endswith('.csv'):
                dataset_name += '.csv'
            file_path = base_dir / dataset_name
        
            self.seqs = []
            possible_seq_cols = ["mutated_sequence", "original_sequence", "seq", "mut_sequence"]
            
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                seq_col = next((col for col in possible_seq_cols if col in reader.fieldnames), None)
                if seq_col is None:
                    raise ValueError(f"无法在文件 {file_path} 中找到序列列，可用列: {reader.fieldnames}")
                
                for row in reader:
                    sequence = row[seq_col].strip()
                    if sequence:   # 可选：忽略空序列
                        self.seqs.append((sequence, 1))
            
            self.label_mapper = dict(enumerate(self.seqs))




                
                
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
