import os
import json
import regex as re
from typing import BinaryIO, Dict, List, Tuple, Iterable, Iterator
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import math # 引入math
import pickle

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
def process_chunk(args: tuple) -> list[tuple[int, ...]]:
    """
    处理单个文本块的辅助函数，用于并行化
    """
    text_chunk, special_tokens, special_tokens_map = args
    return Counter(pre_tokenization(text_chunk, special_tokens, special_tokens_map))   

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
def pre_tokenization(text: str, special_tokens: list[str], special_tokens_map: dict[str, int]) -> list[tuple[int, ...]]:
    """
    对输入的文本进行预分词，返回每个词块的字节ID元组列表
    - 特殊词元：返回 (special_token_id,)
    - 普通文本：返回 (b1, b2, b3, ...)
    """
    if special_tokens is None:
        special_tokens = []

    # 按长度降序排序，避免短词元被误匹配
    sorted_special_tokens = sorted(special_tokens, key=lambda x: -len(x))

    # 构造正则表达式，匹配特殊词元
    special_pattern = "|".join(re.escape(st) for st in sorted_special_tokens)
    if not special_pattern:
        text_parts = [text]
    else:
        text_parts = re.split('(' + special_pattern + ')', text)

    results = []

    for part in text_parts:
        if not part:  # 跳过空字符串
            continue

        if part in special_tokens:
            # 特殊词元：直接映射为 ID
            token_id = special_tokens_map[part]
            results.append((token_id,))
        else:
            # 普通文本：用正则 PAT 预分词，再转为字节ID
            for match in re.finditer(PAT, part):
                word_chunk = match.group(0)
                byte_tuple = tuple(b for b in word_chunk.encode('utf-8'))
                results.append(byte_tuple)

    return results

# 统计频率的辅助函数
def count_chunks(chunks: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    from collections import defaultdict
    freq = defaultdict(int)
    for chunk in chunks:
        freq[chunk] += 1
    return freq

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
                # 核心增量更新
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
            chunk_args.append((text_chunk, special_tokens, special_tokens_map))
    
    # with Pool(num_processes) as pool:
    #     # pool.map现在处理的是包含文本块的参数
    #     list_of_chunks = pool.map(process_chunk, chunk_args)
    
    # # 3. 聚合所有进程的预分词结果
    # print("3. 聚合所有进程的预分词结果...")
    # all_chunks = []
    # for chunks in list_of_chunks:
    #     all_chunks.extend(chunks)
    # word_freqs = count_chunks(all_chunks)
   
    with Pool(num_processes) as pool:
        # list_of_counters 是一个计数字典的列表，而不是巨大词块列表的列表
        list_of_counters = pool.map(process_chunk, chunk_args)
    
    # 3. 高效合并计数字典
    print("3. 聚合所有进程的频率统计结果...")
    word_freqs = Counter()
    for counter in list_of_counters:
        word_freqs.update(counter)

    # 4. 主循环
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

        # 4.2. 直接从stats中找到频率最高的字节对,其中应用了平局字典序最高
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

# 太恶心了,规则不一样居然不说清楚啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
# 我讨厌你我讨厌你我讨厌你
def pre_tokenize(text, vocab, special_tokens, special_tokens_map):
    vocab_reversed = {v: k for k, v in vocab.items()}  # bytes: int
    if special_tokens is None:
        special_tokens = []

    # 按长度降序排序，避免短词元被误匹配
    sorted_special_tokens = sorted(special_tokens, key=lambda x: -len(x))

    # 构造正则表达式，匹配特殊词元
    special_pattern = "|".join(re.escape(st) for st in sorted_special_tokens)
    if not special_pattern:
        text_parts = [text]
    else:
        text_parts = re.split('(' + special_pattern + ')', text)

    results = []

    for part in text_parts:
        if not part:  # 跳过空字符串
            continue

        if part in special_tokens:
            # 特殊词元：直接映射为 ID
            token_id = special_tokens_map[part]
            results.append((token_id,))
        else:
            # 普通文本：用正则 PAT 预分词，再转为字节ID
            for match in re.finditer(PAT, part):
                word_chunk = match.group(0)
                byte_tuple = tuple(vocab_reversed[bytes([b])] for b in word_chunk.encode('utf-8'))
                results.append(byte_tuple)

    return results

####### BPE Tokenizer
# class BPETokenizer:
#     def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
#         self.vocab = vocab
#         self.merges = merges
#         if special_tokens is None:
#             special_tokens = []
#         self.special_tokens = special_tokens 
        
#         self.byte_to_id = {v: k for k, v in vocab.items()}
#         self.special_tokens_map = {st: self.byte_to_id.get(st.encode('utf-8')) for st in self.special_tokens}
        
        
#     @classmethod
#     def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
#         """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges"""
#         # 加载 vocab.pkl
#         with open(vocab_filepath, 'rb') as vf:
#             raw_vocab = pickle.load(vf)
#         # 转换为 {int: bytes}
#         vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
#                 for k, v in raw_vocab.items()}
#         # 加载 merges.pkl
#         with open(merges_filepath, 'rb') as mf:
#             raw_merges = pickle.load(mf)
#         # 转换为 List[Tuple[bytes, bytes]]
#         merges = []
#         for a, b in raw_merges:
#             merges.append((
#                 a.encode("utf-8") if isinstance(a, str) else a,
#                 b.encode("utf-8") if isinstance(b, str) else b
#             ))
#         return cls(vocab, merges, special_tokens)

#     def encode(self, text: str) -> list[int]:
#         """将输入文本编码为词元ID序列"""

#         # 1. 预分词，返回每个词块的字节ID元组列表
#         pre_tokenized_chunks = pre_tokenize(text, self.vocab, self.special_tokens, self.special_tokens_map)
#         pretokens = [list(chunk) for chunk in pre_tokenized_chunks]

#         # 2. 对每个词块分别处理
#         for i, pretoken in enumerate(pretokens):
#             for merge in self.merges:
#                 new_pretoken = []
#                 new_index = self.byte_to_id[merge[0] + merge[1]]
#                 j = 0
#                 while j < len(pretoken):
#                     if (j < len(pretoken)-1) and ((self.vocab[pretoken[j]], self.vocab[pretoken[j+1]]) == merge):
#                         new_pretoken.append(new_index)
#                         j += 2
#                     else:
#                         new_pretoken.append(pretoken[j])
#                         j += 1

#                 pretoken = new_pretoken

#             pretokens[i] = pretoken

#         tokens = [token for pretoken in pretokens for token in pretoken] 
#         return tokens
    
#     def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
#         """Given an iterable of strings (e.g., a Python file handle), 
#         return a generator that lazily yields token IDs. 
#         This is required for memory-eﬀicient tokenization of large files 
#         that we cannot directly load into memory.
#         """
#         for line in iterable:
#             for idx in self.encode(line):
#                 yield idx
    
#     def decode(self, ids: list[int]) -> str:
#         """Decode a sequence of token IDs into text."""
#         tokens = bytes()
#         vocab_size = len(self.vocab)
#         replacement_char = "\uFFFD"

#         for token_id in ids:
#             if token_id < vocab_size:
#                 token = self.vocab[token_id]    # bytes
#             else:
#                 token = bytes(replacement_char, encoding='utf-8')   # Replace tokens with Unicode replacement characters if index out of bounds

#             tokens += token
#         decoded = tokens.decode(encoding='utf-8', errors='replace')

#         return decoded 
    
def to_bytes_tuple(word: str) -> Tuple[bytes]:
    l = list(word.encode("utf-8"))
    l = [bytes([x]) for x in l]
    return tuple(l)    
    
class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        self.merges = merges

        self.bpe_ranks = dict(zip(merges, range(len(merges)))) # 关键!!!!
            
        # Handle special tokens
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        
        # Ensure special tokens are in the vocabulary
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_token_id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.byte_to_token_id[token_bytes] = new_id

    def encode(self, text: str) -> list[int]:
        tokens = []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])
            else:
                tokens.extend(self._tokenize_normal(part))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        # Decode bytes to string, replacing invalid sequences
        return full_bytes.decode("utf-8", errors="replace")

    def _tokenize_normal(self, text: str) -> list[int]:
        """
        Tokenize a normal piece of text (not a special token) into token IDs.
        
        Args:
            text: A string to tokenize.
            
        Returns:
            A list of token IDs representing the tokenized text.
        """
        # Pre-tokenization
        pre_tokens = []
        for m in re.finditer(PAT, text):
            word = m.group(0)
            pre_tokens.append(word)

        token_ids = []
        for token in pre_tokens:
            # Convert token to bytes tuple
            byte_tuple = to_bytes_tuple(token)
            
            # Apply BPE merges
            merged = self._apply_merges(byte_tuple)
            
            # Get token IDs
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        
        return token_ids

    def _apply_merges(self, byte_tuple: tuple[bytes, ...]) -> list[bytes]:
        """
        Apply BPE merges to a sequence of bytes.
        
        Args:
            byte_tuple: A tuple of single-byte tokens.
            
        Returns:
            A list of merged byte tokens after applying all applicable merges.
        """
        word: list[bytes] = list(byte_tuple)

        def get_pairs(word: list[bytes]):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs
        
        pairs = get_pairs(word)

        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        return word    