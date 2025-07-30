import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.BPETokenizer import bpe_train
import pathlib

if __name__ == '__main__':
    
    # 获取 train_bpe.py 文件所在的目录 (即 .../CS336-LLM/train)
    HERE = pathlib.Path(__file__).resolve().parent

    # 从 train 目录回到上一级项目根目录 (.../CS336-LLM)
    PROJECT_ROOT = HERE.parent 

    # 构建到数据文件的绝对路径
    data_path = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    bpe_train(input_path=data_path, vocab_size=vocab_size, special_tokens=special_tokens)