"""
python model/generate.py \
  --prompt "In a land of myths, and a time of magic," \
  --checkpoint_iter 6000 \
  --max_new_tokens 200 \
  --temperature 0.7 \
  --top_k 100--prompt "The little bear said," --checkpoint_iter 5000

"""

import os
import sys
import json
import torch
import pickle
import pathlib
import argparse

# --- 1. 项目路径设置 ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "train/config.json" # 配置文件路径

# 从你的代码库中导入必要的模块
from model.Transformer import TransformerLM
from model.BPETokenizer import BPETokenizer 


def main():
    """
    模型推理主函数
    """
    # --- 2. 加载配置文件 ---
    print(f"正在从 '{CONFIG_PATH}' 加载配置...")
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{CONFIG_PATH}' 未找到。")
        sys.exit(1)

    # --- 3. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="使用 TransformerLM 模型进行文本生成")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="用于生成文本的输入提示")
    # 让用户可以指定使用哪个迭代次数的检查点
    parser.add_argument("--checkpoint_iter", type=int, default=config['lm_params']['training']['train_steps'], help="要加载的模型检查点的迭代次数")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="要生成的最大新词元数量")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k 采样")
    args = parser.parse_args()

    # --- 4. 设备和编译设置 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")

    # --- 5. 加载 Tokenizer (路径来自 config) ---
    bpe_params = config['bpe_params']
    # 假设分词器文件存放在 'PROJECT_ROOT/tokenizer/' 目录下
    tokenizer_dir = PROJECT_ROOT / "data"
    vocab_path = tokenizer_dir / bpe_params['vocab_file']
    merges_path = tokenizer_dir / bpe_params['merges_file']
    special_tokens = bpe_params['special_tokens']

    print("正在加载 Tokenizer...")
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)
        tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
        eos_token_id = tokenizer.byte_to_token_id.get("<|endoftext|>".encode("utf-8"))
    except FileNotFoundError:
        print(f"错误: Tokenizer 文件在 '{tokenizer_dir}' 中未找到。请检查 config.json 中的文件名和实际路径。")
        sys.exit(1)

    # --- 6. 加载模型 (结构和权重) ---
    # 从 config 加载模型结构参数
    model_config = config['lm_params']['model']
    model = TransformerLM(**model_config)
    model.to(device)
    
    # 从 config 构建检查点路径并加载权重
    ckpt_dir = PROJECT_ROOT / config['output_paths']['checkpoints_dir']
    ckpt_file = ckpt_dir / f"ckpt_iter{args.checkpoint_iter}.pt"
    
    print(f"正在从 '{ckpt_file}' 加载模型权重...")
    try:
        # in generate.py, line 96
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
        # 兼容 torch.compile 编译后模型保存的 state_dict
        state_dict = checkpoint['model_state_dict']
        unwanted_prefix = '_orig_mod.'
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"错误: 模型检查点 '{ckpt_file}' 未找到。请确保已完成训练或指定了正确的迭代次数。")
        sys.exit(1)

    model.eval()
    if device == 'cuda':
        print("正在编译模型 (这可能需要一些时间)...")
        model = torch.compile(model)
    
    # --- 7. 编码输入并生成 ---
    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print("\n" + "="*20 + " 生成中 " + "="*20)
    print(f"输入: '{args.prompt}'")
    
    # 调用模型的 generate 方法
    output_tokens = model.generate(
        x=input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_token_id
    )
    
    # 提取生成的 token ID
    output_ids = output_tokens[0].cpu().tolist()
    
    # --- 8. 解码并打印结果 ---
    generated_text = tokenizer.decode(output_ids)
    
    print("\n" + "="*20 + " 生成结果 " + "="*20)
    print(args.prompt + generated_text)
    print("="*50 + "\n")


if __name__ == "__main__":
    main()