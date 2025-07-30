# My-Transformer-LM
2025/7/20 开始stanford cs336课程

2025/7/24 完成BPETokenizer,通过测试

2025/7/28 完成搭建模型所需全部模块并组装完成TransformerLM

2025/7/30 完成训练代码,准备开始训练


## Download data
Download the TinyStories data and a subsample of OpenWebText
```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
