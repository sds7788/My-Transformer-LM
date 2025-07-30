import sys
import os
import json
import pathlib
import torch
import numpy as np
from tqdm import tqdm
import wandb

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.Transformer import TransformerLM
from model.Loss import cross_entropy_loss
from model.AdamW import AdamW
from model.GradientClipping import gradient_clipping
from model.lrSchedule import get_lr_cosine_schedule
from model.CheckPoints import save_checkpoint, load_checkpoint

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "train/config.json" 


def _to_device_and_compile(model):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"正在使用设备: {device}")
    model = model.to(device)

    if device.type == "cuda":
        print("正在编译模型 (backend='inductor')...")
        model = torch.compile(model)
    else: 
        print("未找到device")
   
    return model, device

def get_memmap_dataset(path: str, dtype=np.uint16):
    """以内存映射方式加载 tokenized 数据集。"""
    return np.memmap(path, dtype=dtype, mode="r")

def get_batch(memmap_arr, batch_size, context_length, device):
    """从数据集中随机采样一个批次。"""
    N = len(memmap_arr)
    # 随机选择起始点
    ix = np.random.randint(0, N - context_length, size=(batch_size,))
    # 构建 x 和 y
    x = np.stack([memmap_arr[i : i + context_length] for i in ix])
    y = np.stack([memmap_arr[i + 1 : i + context_length + 1] for i in ix])
    # 转换为 Tensor 并移动到设备
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)

def memmap_val_iterator(memmap_arr, batch_size, context_length, device):
    """为验证集创建一个迭代器。"""
    N = len(memmap_arr)
    num_batches = (N - 1) // (batch_size * context_length)
    for i in range(num_batches):
        start_idx = i * batch_size * context_length
        end_idx = start_idx + batch_size * context_length
        x_block = memmap_arr[start_idx : end_idx]
        y_block = memmap_arr[start_idx + 1 : end_idx + 1]
        x = torch.from_numpy(x_block.reshape(batch_size, context_length)).long().to(device)
        y = torch.from_numpy(y_block.reshape(batch_size, context_length)).long().to(device)
        yield x, y

def main():
    # 加载配置文件
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 将 model 和 training 的配置合并到一个对象中，方便访问
    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = DotDict(config['lm_params']['training'])
    args.update(config['lm_params']['model'])
    
    # 初始化 wandb 用于实验跟踪
    wandb.init(project="transformer-training-project", config=args)
    
    # 使用配置创建模型并移动到设备
    model = TransformerLM(**config['lm_params']['model'])
    model, device = _to_device_and_compile(model)
    
    # 确保检查点目录存在
    save_dir_name = config['output_paths']['checkpoints_dir']
    SAVE_DIR = PROJECT_ROOT / save_dir_name
    SAVE_DIR.mkdir(exist_ok=True)
    
    # 加载数据集，路径来自配置
    train_data_path = DATA_DIR / config['data_paths']['tokenized_train']
    val_data_path = DATA_DIR / config['data_paths']['tokenized_valid']
    train_data = get_memmap_dataset(str(train_data_path))
    val_data = get_memmap_dataset(str(val_data_path))
    
    # 构建优化器
    optimizer = AdamW()(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 恢复断点
    start_iter = 0
    if args.resume_checkpoint > 0:
        resume_ckpt_path = SAVE_DIR / f"ckpt_iter{args.resume_checkpoint}.pt"
        if resume_ckpt_path.exists():
            print(f"正在从检查点恢复: {resume_ckpt_path}")
            start_iter = load_checkpoint(resume_ckpt_path, model, optimizer)
            print(f"已恢复至迭代次数: {start_iter}")
        else:
            print(f"警告: 检查点 {resume_ckpt_path} 未找到。将从头开始训练。")
    
    # 训练循环
    for iteration in tqdm(range(start_iter, args.train_steps), desc="Training"):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.shape[-1]), y.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.clip_grad_norm)
        
        # 更新学习率
        lr = get_lr_cosine_schedule(iteration, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # 日志记录
        if (iteration + 1) % 10 == 0:
            wandb.log({"train_loss": loss.item(), "learning_rate": lr}, step=iteration + 1)
                  
        # 验证
        if (iteration + 1) % args.val_interval == 0:
            model.eval()
            val_loss_agg = 0.0
            val_batch_count = 0
            with torch.no_grad():
                val_iter = memmap_val_iterator(val_data, args.batch_size, args.context_length, device)
                for x_val, y_val in val_iter:
                    if val_batch_count >= args.val_batches:
                        break
                    val_logits = model(x_val)
                    val_loss = cross_entropy_loss(val_logits.view(-1, val_logits.shape[-1]), y_val.view(-1))
                    val_loss_agg += val_loss.item()
                    val_batch_count += 1
            
            val_loss_mean = val_loss_agg / val_batch_count
            print(f"\n迭代 {iteration+1:05d}: 验证集损失 = {val_loss_mean:.4f}")
            wandb.log({"val_loss": val_loss_mean}, step=iteration + 1)

        # 保存检查点
        if (iteration + 1) % args.save_interval == 0:
            ckpt_path = SAVE_DIR / f"ckpt_iter{iteration+1}.pt"
            save_checkpoint(model, optimizer, iteration + 1, ckpt_path)
            print(f"检查点已保存至: {ckpt_path}")
            
    wandb.finish() # 结束 wandb run
    
if __name__ == "__main__":
    main()