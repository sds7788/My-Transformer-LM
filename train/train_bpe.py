import sys
import json
import pathlib
import pickle
import numpy as np
import time
from typing import Iterable

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

    vocab, merges = bpe_train(
        input_path=str(raw_train_path), # 确保传入的是字符串路径
        vocab_size=bpe_cfg['vocab_size'],
        special_tokens=bpe_cfg['special_tokens'],
    )
    
    save_pkl(vocab, vocab_path)
    save_pkl(merges, merges_path)

    duration = time.time() - start_time
    print(f"BPE 训练完成，耗时: {duration // 60:.0f} 分 {duration % 60:.0f} 秒。\n")


    print("--- 步骤 2: 开始数据 Tokenization ---")
    
    tokenizer = BPETokenizer.from_files(str(vocab_path), str(merges_path), bpe_cfg['special_tokens'])

    # 处理训练集
    train_file_path = DATA_DIR / data_cfg['raw_train']
    train_encode_path = DATA_DIR / data_cfg['tokenized_train']
    print(f"正在处理训练集: {train_file_path} -> {train_encode_path}")
    with open(train_file_path, 'r', encoding='utf-8') as f:
        save_encode_stream(tokenizer.encode_iterable(f), train_encode_path)

    # 处理验证集
    valid_file_path = DATA_DIR / data_cfg['raw_valid']
    valid_encode_path = DATA_DIR / data_cfg['tokenized_valid']
    print(f"正在处理验证集: {valid_file_path} -> {valid_encode_path}")
    with open(valid_file_path, 'r', encoding='utf-8') as f:
        save_encode_stream(tokenizer.encode_iterable(f), valid_encode_path)
        
    print("数据 Tokenization 完成。")

if __name__ == '__main__':
    main()