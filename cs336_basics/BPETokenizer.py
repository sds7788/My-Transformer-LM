import os
import regex as re
import time
import json
import psutil
import requests
from typing import BinaryIO, Dict, List, Tuple
from collections import defaultdict
from multiprocessing import Pool, cpu_count

##### BPE-trainer

# github仓库提供的边界函数
def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    Args：
        file：二进制文件对象
        desired_num_chunks:希望分成多少块
        split_special_token:用于分割的特殊字节

    Returns:
        List[int]:一个整数列表,每个元素是文件的分块边界
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 并行化预分词辅助处理函数
# 并行处理的辅助函数
def process_chunk(args: tuple) -> dict[tuple[int, ...], int]:
    """
    处理单个文本块的辅助函数，用于并行化
    """
    # 直接接收文本块和 special_tokens_map
    text_chunk, special_tokens_map = args
    return pre_tokenization(text_chunk, special_tokens_map)


# 词表初始化函数
def init_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], dict[str,int]]:
    """
    初始化词汇表：

    Args:
        special_tokens(list[str]): 一个特殊词元的字符串列表

    Returns：
        tuple[dict[int, bytes], dict[str, int]]
            - vocab:ID到字节的映射
            - special_tokens:特殊词元字符串到ID的映射

    """
    # vocab是我们最终要返回的词汇表，用于解码(ID->词元)

    # 首先基础词汇表要包含256个字节
    vocab = {i: bytes([i]) for i in range(256)}

    # 添加特殊词元，从256开始为每个特殊词元分配ID
    # 设立一个special_tokens_map记录ID和特殊词元的映射关系
    special_tokens_map = {}

    for i, token_str in enumerate(special_tokens):
        token_id = i + 256
        vocab[token_id] = token_str.encode("utf-8")
        special_tokens_map[token_str] = token_id

    return vocab, special_tokens_map


# 预分词所使用的正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 预分词函数
def pre_tokenization(text: str, special_tokens_map: dict[str, int]) -> dict[tuple[int, ...], int]:
    """
    对输入的文本进行预分词，统计预分词后个单词块的频率

    Args:
        text：输入文本
        special_tokens_map:特殊词元及映射

    Returns:
        dict[tuple[int, ...], int]:一个字典，键是表示单词块的字节ID元组，值是出现的频率
    """
    word_freqs = defaultdict(int)

    # 使用特殊词元来分割文本，确保他们不参与BPE合并
    special_pattern = "|".join(re.escape(st) for st in special_tokens_map)

    if not special_pattern:
        text_parts = [text]
    else:
        text_parts = re.split(f"({special_pattern})", text)

    for part in text_parts:
        if part in special_tokens_map:
            continue
        
        for match in re.finditer(PAT, part):
            word_chunk = match.group(0)
            byte_tuple = tuple(b for b in word_chunk.encode('utf-8'))
            word_freqs[byte_tuple] += 1
            
    return word_freqs

# 合并函数
# 首先我们计算所有相邻字节对的频率
def get_pair_stats(word_freqs: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """ 从单词频率中计算所有相邻字节对的频率 """
    pair_stats = defaultdict(int)
    # 遍历每一个单词块及其出现频率
    for word_tuple, freq in word_freqs.items():
        # 在单词块内部，遍历所有相邻的字节对
        for i in range(len(word_tuple) - 1):
            # 获取一个字节对，例如 (116, 104) 代表 ('t', 'h')
            pair = (word_tuple[i], word_tuple[i+1])
            # 将这个字节对的计数增加单词块出现的次数
            pair_stats[pair] += freq
    return pair_stats

### 太过于低效,选择效率更高的数据结构,采用增量更新
# def merge_pair(word_freqs: dict[tuple[int, ...], int], pair_to_merge: tuple[int, int], new_token_id: int) -> dict[tuple[int, ...], int]:
#     """ 在所有单词块中合并指定的字节对 """
#     new_word_freqs = defaultdict(int)
#     p1, p2 = pair_to_merge
    
#     # 遍历每一个单词块
#     for word_tuple, freq in word_freqs.items():
#         new_word_tuple = []
#         i = 0
#         while i < len(word_tuple):
#             # 检查当前位置和下一位置是否构成了我们要合并的对
#             if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i+1] == p2:
#                 # 如果是，就用新的ID替换这个对
#                 new_word_tuple.append(new_token_id)
#                 i += 2  # 指针向前移动两位
#             else:
#                 # 如果不是，就保留原来的字节ID
#                 new_word_tuple.append(word_tuple[i])
#                 i += 1  # 指针向前移动一位
#         # 将新生成的单词块（元组形式）及其原始频率存入新字典
#         new_word_freqs[tuple(new_word_tuple)] += freq
        
#     return new_word_freqs

# 合并和增量更新函数
def merge_and_update_stats(
    word_freqs: dict[tuple[int, ...], int],
    pair_to_merge: tuple[int, int],
    new_token_id: int,
    stats: dict[tuple[int, int], int]
) -> dict[tuple[int, ...], int]:
    """
    在所有单词块中合并指定的字节对，并以增量方式高效更新字节对频率统计。

    Args:
        word_freqs: 当前的单词块及其频率。
        pair_to_merge: 要合并的字节对 (p1, p2)。
        new_token_id: 合并后产生的新词元ID。
        stats: 需要被增量更新的全局字节对频率统计字典。

    Returns:
        合并后的新单词块频率字典。
    """
    new_word_freqs = defaultdict(int)
    p1, p2 = pair_to_merge

    for word_tuple, freq in word_freqs.items():
        # 如果单词块中没有要合并的对，直接跳过，无需处理
        if len(word_tuple) < 2:
            new_word_freqs[word_tuple] += freq
            continue

        new_word_tuple = []
        i = 0
        merged = False
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i+1] == p2:
                # --- 核心增量更新逻辑 ---
                # 1. 更新被破坏的旧邻居关系
                if i > 0:
                    prev_token = word_tuple[i-1]
                    stats[(prev_token, p1)] -= freq
                    # 2. 创建并更新新邻居关系
                    stats[(prev_token, new_token_id)] += freq
                
                # 1. 更新被破坏的旧邻居关系 (后一个邻居)
                if i < len(word_tuple) - 2:
                    next_token = word_tuple[i+2]
                    stats[(p2, next_token)] -= freq
                    # 2. 创建并更新新邻居关系
                    stats[(new_token_id, next_token)] += freq

                new_word_tuple.append(new_token_id)
                i += 2
                merged = True
            else:
                new_word_tuple.append(word_tuple[i])
                i += 1
        
        # 如果发生了合并，才用新的元组，否则用旧的
        if merged:
            new_word_freqs[tuple(new_word_tuple)] += freq
        else:
            new_word_freqs[word_tuple] += freq

    return new_word_freqs

# BPE train函数
def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    BPE分词器训练函数

    Args:
        input_path:包含BPE分词器训练数据的文本文件路径
        vocab_size:定义最终词汇表的大小
        special_tokens:特殊词元

    Returns:
        vocab:最终词汇表
        merges:训练时产生的BPE合并列表
    """

    # 1. 初始化词汇表
    print("1. 初始化词汇表...")
    vocab, special_tokens_map = init_vocab(special_tokens)

    # 2. 并行预分词
    print("2. 进行并行预分词...")
    num_processes = cpu_count()
    print(f"将使用{num_processes}个进程并行预分词")
    
    chunk_args = []
    with open(input_path, "rb") as f:  # 以二进制模式打开
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        print(f"文件被分割成 {len(boundaries) - 1} 个块。")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            text_chunk = f.read(end - start).decode("utf-8", "ignore").replace('\r\n', '\n') # 不同操作系统的换行符问题
            # 准备好解码后的文本块作为参数
            chunk_args.append((text_chunk, special_tokens_map))
    
    with Pool(num_processes) as pool:
        # pool.map现在处理的是包含文本块的参数
        list_of_word_freqs = pool.map(process_chunk, chunk_args)
    
    # 3. 聚合所有进程的预分词结果
    print("3. 聚合所有进程的预分词结果...")
    word_freqs = defaultdict(int)
    for single_word_freqs in list_of_word_freqs:
        for word, freq in single_word_freqs.items():
            word_freqs[word] += freq
            
    # # 4. BPE主循环
    # print("4. 开始BPE合并训练...")
    # merges = []
    # num_merges = vocab_size - len(vocab) # 需要执行的合并次数

    # for i in range(num_merges):
    #     # 4.1. 计算字节对频率
    #     pair_stats = get_pair_stats(word_freqs)

    #     # 如果没有更多可合并的对，提前结束
    #     if not pair_stats:
    #         print("没有更多可合并的字节对，训练提前结束。")
    #         break

    #     # 4.2. 找到频率最高的字节对
    #     best_pair = max(
    #         pair_stats.keys(),
    #         key=lambda p: (
    #             pair_stats[p],
    #             vocab.get(p[0], b'').decode('utf-8', 'replace'),
    #             vocab.get(p[1], b'').decode('utf-8', 'replace')
    #         )
    #     )

    #     # 4.3. 创建新的词元ID
    #     new_token_id = len(vocab)

    #     # 4.4. 在词块中合并该字节对
    #     word_freqs = merge_pair(word_freqs, best_pair, new_token_id)
        
    #     # 4.5. 记录合并信息 (以字节形式)
    #     p1, p2 = best_pair
    #     b1 = vocab[p1]
    #     b2 = vocab[p2]
    #     merges.append((b1, b2))
        
    #     # 4.6. 更新词汇表
    #     vocab[new_token_id] = b1 + b2

    #     # 打印进度
    #     if (i + 1) % 50 == 0 or i == num_merges - 1:
    #         # 使用repr来显示字节串，避免乱码
    #         print(f"合并 {i+1}/{num_merges}: {best_pair} -> {new_token_id} (新词元: {repr(vocab[new_token_id])})")

    # print("\nBPE训练完成！")
    # print(f"最终词汇表大小: {len(vocab)}")

    # # 5. 返回结果
    # return vocab, merges

    # 4. BPE主循环
    print("4. 开始BPE合并训练...")
    merges = []
    num_merges = vocab_size - len(vocab)

    # 4.1. 一次性初始化字节对频率统计
    print("   进行一次性字节对频率初始化...")
    pair_stats = get_pair_stats(word_freqs)

    for i in range(num_merges):
        # 如果没有更多可合并的对，提前结束
        if not pair_stats:
            print("没有更多可合并的字节对，训练提前结束。")
            break

        # 4.2. 直接从stats中找到频率最高的字节对
        best_pair = max(
            pair_stats.keys(),
            key=lambda p: (
                pair_stats[p],
                vocab.get(p[0], b'').decode('utf-8', 'replace'),
                vocab.get(p[1], b'').decode('utf-8', 'replace')
            )
        )

        # 4.3. 创建新的词元ID
        new_token_id = len(vocab)
        
        # 4.4. 调用新的合并与增量更新函数
        word_freqs = merge_and_update_stats(word_freqs, best_pair, new_token_id, pair_stats)
        
        # 4.5. 从 stats 中移除已经合并的旧字节对
        pair_stats.pop(best_pair)

        # 4.6. 记录合并信息
        p1, p2 = best_pair
        merges.append((vocab[p1], vocab[p2]))
        
        # 4.7. 更新词汇表
        vocab[new_token_id] = vocab[p1] + vocab[p2]

        # 打印进度
        if (i + 1) % 50 == 0 or i == num_merges - 1:
            print(f"合并 {i+1}/{num_merges}: {best_pair} -> {new_token_id} (新词元: {repr(vocab[new_token_id])})")

    print("\nBPE训练完成！")
    print(f"最终词汇表大小: {len(vocab)}")
    
    return vocab, merges

####### 为训练让Gemini写的一个主程序
def save_results(vocab: dict, merges: list):
    """将词汇表和合并规则保存到磁盘"""
    # 保存词汇表
    # 将字节值转换为Base64编码的字符串以便JSON序列化
    import base64
    vocab_serializable = {
        token_id: base64.b64encode(byte_val).decode('utf-8') 
        for token_id, byte_val in vocab.items()
    }
    vocab_path = "tinystories_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab_serializable, f, indent=2)
    print(f"词汇表已保存至: {vocab_path}")

    # 保存合并规则
    merges_path = "tinystories_merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # repr()可以很好地处理字节串的表示
            f.write(f"{repr(p1)} {repr(p2)}\n")
    print(f"合并规则已保存至: {merges_path}")

if __name__ == "__main__":
    # --- (a) 在TinyStories上训练BPE ---
    
    # 记录初始内存
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2) # MB

    # 定义训练参数
    INPUT_FILE_PATH = "../data/TinyStoriesV2-GPT4-train.txt" 
    VOCAB_SIZE = 10000
    SPECIAL_TOKENS = ["<|endoftext|>"]
    
    # 开始计时
    start_time = time.time()

    # 执行训练
    vocab, merges, timings = bpe_train(
        input_path=INPUT_FILE_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )
    
    # 结束计时
    end_time = time.time()
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60

    # 记录峰值内存
    mem_after = process.memory_info().rss / (1024 ** 2) # MB
    peak_memory_mb = max(mem_before, mem_after) # 简易峰值估算

    # 保存结果
    save_results(vocab, merges)
    
    print("\n--- 训练结果分析 ---")
    
    # (a) 问题回答
    print("(a) 训练耗时与资源使用情况:")
    print(f"训练总耗时: {total_time_minutes:.2f} 分钟 ({total_time_seconds:.2f} 秒)")
    print(f"峰值内存使用估算: {peak_memory_mb:.2f} MB")
    
    # 找出最长的词元
    longest_token_len = 0
    longest_token_val = b''
    for token_bytes in vocab.values():
        if len(token_bytes) > longest_token_len:
            longest_token_len = len(token_bytes)
            longest_token_val = token_bytes

    print(f"\n词汇表中最长的词元是: {repr(longest_token_val)}")
    print(f"它的长度是: {longest_token_len} 字节")
    # 尝试解码并分析意义
    try:
        decoded_token = longest_token_val.decode('utf-8')
        print(f"解码后的内容是: '{decoded_token}'")
        print("它是否有意义？ 对于TinyStories这种儿童故事集，出现像 ' once upon a time' 或 ' and they lived happily' 这样反复出现的长短语是完全合理的。")
    except UnicodeDecodeError:
        print("它无法被完整解码为UTF-8字符串，说明它可能是一个跨越多个字符边界的字节序列，但这在BPE中是正常的。")

    print("\n" + "="*40 + "\n")

    # --- (b) 性能分析 ---
    print("(b) 训练过程各部分耗时分析:")
    for stage, duration in timings.items():
        print(f" - {stage:<30}: {duration:.4f} 秒")

    slowest_stage = max(timings, key=timings.get)
    print(f"\n耗时最长的部分是: '{slowest_stage}'，耗时 {timings[slowest_stage]:.2f} 秒。")