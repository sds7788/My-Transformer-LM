import sys
import json
import pathlib
import pickle
import numpy as np
import time
from typing import Iterable
import os
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.BPETokenizer import BPETokenizer, bpe_train

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "train/config.json"

def save_pkl(file, file_path: pathlib.Path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def load_pkl(file_path: pathlib.Path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
        return file

def save_encode_stream(token_stream: Iterable[int], file_path: pathlib.Path):
    array = np.fromiter(token_stream, dtype=np.uint16)
    array.tofile(file_path)
    
def worker_tokenize_chunk(args: tuple) -> list[int]:
    """
    这是一个“工作函数”，由每个子进程独立执行。
    它接收一个文件块的范围，读取、编码并返回结果。
    """
    tokenizer, file_path, start_byte, end_byte = args
    
    # 每个子进程都重新打开一次文件，是安全的操作
    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(start_byte)  # 定位到块的起始位置
        text_chunk = f.read(end_byte - start_byte) # 读取指定块的内容
        return tokenizer.encode(text_chunk) # 使用传入的tokenizer进行编码
    
# --- 新增的并行处理主函数 ---
def parallel_tokenize(tokenizer: BPETokenizer, input_path: str, output_path: str, num_processes: int = None):
    """
    使用多进程并行地对一个文本文件进行分词，并将结果保存。
    """
    if num_processes is None:
        num_processes = cpu_count() # 默认使用所有CPU核心

    print(f"正在使用 {num_processes} 个进程并行处理 {input_path}...")

    # 1. 智能文件分块
    chunk_boundaries = [0]
    file_size = os.path.getsize(input_path)
    chunk_size = file_size // num_processes
    
    with open(input_path, "rb") as f:  # 必须用二进制模式'rb'来精确定位
        for i in range(1, num_processes):
            seek_pos = chunk_size * i
            f.seek(seek_pos)
            f.readline()  # 读取到下一个换行符，确保不会切断一行
            boundary = f.tell()
            chunk_boundaries.append(boundary)
    chunk_boundaries.append(file_size)

    # 2. 准备每个进程的任务参数
    tasks = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        if start >= end: continue
        tasks.append((tokenizer, input_path, start, end))

    # 3. 创建进程池并分发任务
    all_tokens = []
    with Pool(num_processes) as pool:
        # imap 会按顺序返回结果，非常适合文件处理
        # 它返回一个迭代器，可以节省主进程的内存
        for token_chunk in pool.imap(worker_tokenize_chunk, tasks, chunksize=1):
            all_tokens.extend(token_chunk)
    
    # 4. 保存结果
    save_encode_stream(iter(all_tokens), output_path)        

def main():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    data_cfg = config['data_paths']
    bpe_cfg = config['bpe_params']

    print("--- 步骤 1: 开始 BPE 训练 ---")
    start_time = time.time()

    raw_train_path = DATA_DIR / data_cfg['raw_train']
    vocab_path = DATA_DIR / bpe_cfg['vocab_file']
    merges_path = DATA_DIR / bpe_cfg['merges_file']

    if vocab_path.exists() and merges_path.exists():
        # 如果文件已存在，则跳过训练
        print("--- 步骤 1: BPE 训练已跳过 ---")
        print(f"原因: 词汇表和合并规则文件已存在。")
        print(f"  - 词汇表: {vocab_path}")
        print(f"  - 合并规则: {merges_path}\n")
    else:
        # 如果文件不存在，则执行训练
        print("--- 步骤 1: 开始 BPE 训练 ---")
        start_time = time.time()
        
        vocab, merges = bpe_train(
            input_path=str(raw_train_path), 
            vocab_size=bpe_cfg['vocab_size'],
            special_tokens=bpe_cfg['special_tokens'],
        )
        
        print(f"训练完成，正在保存文件...")
        save_pkl(vocab, vocab_path)
        save_pkl(merges, merges_path)

        duration = time.time() - start_time
        print(f"BPE 训练及保存完成，耗时: {duration // 60:.0f} 分 {duration % 60:.0f} 秒。\n")


    print("--- 步骤 2: 开始数据 Tokenization ---")
    # 读取词表和merges
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)

    # 构造tokenizer
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=bpe_cfg['special_tokens']
    )
        
    # # 处理训练集
    # train_file_path = DATA_DIR / data_cfg['raw_train']
    # train_encode_path = DATA_DIR / data_cfg['tokenized_train']
    # print(f"正在处理训练集: {train_file_path} -> {train_encode_path}")
    # with open(train_file_path, 'r', encoding='utf-8') as f:
    #     save_encode_stream(tokenizer.encode_iterable(f), train_encode_path)

    # # 处理验证集
    # valid_file_path = DATA_DIR / data_cfg['raw_valid']
    # valid_encode_path = DATA_DIR / data_cfg['tokenized_valid']
    # print(f"正在处理验证集: {valid_file_path} -> {valid_encode_path}")
    # with open(valid_file_path, 'r', encoding='utf-8') as f:
    #     save_encode_stream(tokenizer.encode_iterable(f), valid_encode_path)
    # 并行处理训练集
    train_file_path = DATA_DIR / data_cfg['raw_train']
    train_encode_path = DATA_DIR / data_cfg['tokenized_train']
    parallel_tokenize(tokenizer, str(train_file_path), train_encode_path)

    # 并行处理验证集
    valid_file_path = DATA_DIR / data_cfg['raw_valid']
    valid_encode_path = DATA_DIR / data_cfg['tokenized_valid']
    parallel_tokenize(tokenizer, str(valid_file_path), valid_encode_path)
        
    print("数据 Tokenization 完成。")

if __name__ == '__main__':
    main()