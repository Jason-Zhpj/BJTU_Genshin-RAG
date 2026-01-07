# ⚡ GenshinRAG: 基于 FlashRAG 的原神垂直领域问答助手

本项目基于 [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) 框架，实现了一个针对《原神》游戏知识的垂直领域 RAG（检索增强生成）系统。

## 1. 环境配置 (Environment Setup)

首先，建议创建一个新的 Conda 环境以避免依赖冲突。本项目需要 Python 3.10 或更高版本。

Bash

```
# 1. 创建并激活 conda 环境
conda create -n genshin_rag python=3.10
conda activate genshin_rag

# 2. 安装 FlashRAG (开发版)
pip install flashrag-dev --pre

# 或者从源码安装
# git clone https://github.com/RUC-NLPIR/FlashRAG.git
# cd FlashRAG
# pip install -e .

# 3. 安装项目依赖
# 我们的 demo_zh.py 依赖 streamlit, jieba 以及 bm25s 等库
pip install -r requirements.txt
```

> **注意**: 如果你计划使用 vLLM 进行加速推理，请确保安装了兼容的 CUDA 版本并执行 `pip install vllm`。

## 2. 模型与数据准备 (Model & Data Preparation)

本项目包含两类模型：用于生成的 **LLM (大语言模型)** 和用于检索的 **Retriever (检索模型)**（如果是稠密检索）。

### 2.1 下载模型

你需要下载以下模型权重（或使用你自己的模型路径）：

1. **生成模型 (Generator)**: 本项目默认使用 **Qwen1.5-14B-Chat**。
   - 下载地址: [HuggingFace](https://huggingface.co/Qwen/Qwen1.5-14B-Chat) 或 [ModelScope](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat).
   - 存放路径示例: `/data/models/Qwen1.5-14B-Chat`
2. **检索模型 (Retriever) [可选]**: 如果你使用的是 `bm25`，则不需要下载模型权重。如果你想使用稠密检索（如 `e5`），则需要下载对应模型。
   - 下载地址: [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
   - 存放路径示例: `/data/models/e5-base-v2`

### 2.2 准备语料库 (`genshin.jsonl`)

确保你的知识库文件 `genshin_v2.jsonl` 格式符合 FlashRAG 要求。每行应为一个 JSON 对象，必须包含 `"id"` 和 `"contents"` 字段。

**数据格式示例**:

代码段

```
{"id": "001", "contents": "标题：角色A攻略\n内容：角色A是一个强力的火属性输出角色...", "title": "角色A攻略"}
{"id": "002", "contents": "标题：秘境B打法\n内容：秘境B推荐使用雷元素角色...", "title": "秘境B打法"}
```

## 3. 构建索引 (Index Construction)

在进行推理之前，我们需要基于 `genshin.jsonl` 构建检索索引。本项目支持 BM25（稀疏检索）和 E5（稠密检索）。

### 方案 A: 构建 BM25 索引 (推荐，与 Demo 代码一致)

`demo_zh.py` 中默认使用的是 `bm25s` 后端。使用以下命令构建索引：

Bash

```
# 假设你的语料库在 /data/corpus/genshin_v2.jsonl
# 索引将保存到 /data/indexes/bm25

python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path /data/corpus/genshin_v2.jsonl \
    --bm25_backend bm25s \
    --save_dir /data/indexes/bm25
```

### 方案 B: 构建 E5 稠密索引 (可选)

如果你想获得更好的语义检索效果，可以使用 `scripts/build_index.sh` 构建稠密索引：

Bash

```
# 修改 build_index.sh 中的路径或直接运行以下命令
# 注意：需要 GPU 支持

CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /data/models/e5-base-v2 \
    --corpus_path /data/corpus/genshin_v2.jsonl \
    --save_dir /data/indexes/genshin_e5 \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method mean \
    --faiss_type Flat
```

## 4. 运行推理 Demo (Inference)

我们将使用 `demo_zh.py` 启动一个 Streamlit Web 界面来进行问答。

### 4.1 修改配置文件

打开 `demo_zh.py`，找到 `config_dict` 部分。**你必须修改其中的路径为你本地的实际路径**。

Python

```
# demo_zh.py

config_dict = {
    "save_note": "demo",
    "generator_model": "qwen-14B",
    # 【修改点1】LLM 模型路径
    "model2path": {"qwen-14B": "/your/local/path/to/Qwen1.5-14B-Chat"}, 
    
    # 【修改点2】语料库路径
    "corpus_path": "/your/local/path/to/genshin_v2.jsonl",
    
    # 【修改点3】检索方式 (如果你构建了 E5 索引，这里改为 "e5")
    "retrieval_method": "bm25", 
    
    # 【修改点4】索引保存目录 (对应第3步中 --save_dir 的路径)
    "index_path": "/your/local/path/to/indexes/bm25",
    
    "bm25_backend": "bm25s",
}
```

### 4.2 启动 WebUI

在终端中运行以下命令启动服务：

Bash

```
streamlit run demo_zh.py
```

### 4.3 使用指南

1. **浏览器访问**: 启动成功后，浏览器会自动打开（通常是 `http://localhost:8501`）。
2. **侧边栏配置**:
   - **Temperature**: 控制生成的随机性。
   - **Top-K**: 检索的文档数量（推荐 3-5）。
   - **Max Tokens**: 生成回答的最大长度。
3. **输入问题**: 在文本框中输入关于原神的问题，例如：“胡桃的圣遗物推荐什么？”
4. **生成结果**: 点击 "Generate Responses" 按钮。
   - 系统会展示 **References**（检索到的原始文档片段）。
   - 展示 **Response with RAG**（结合文档生成的回答）。
   - 展示 **Response without RAG**（模型仅凭自身训练知识的回答，用于对比）。

------

### 常见问题 (FAQ)

- **报错 `Index not found`**: 请检查 `config_dict` 中的 `index_path` 是否正确指向了包含 `.index` 文件（或 bm25 文件夹）的目录。
- **BM25s 报错**: 确保安装了 `bm25s` (`pip install bm25s`).
- **显存不足**: 如果运行 14B 模型显存不足，可以尝试在 `config_dict` 中添加 `"generator_load_kwargs": {"load_in_8bit": True}` (需要安装 bitsandbytes) 或更换更小的模型（如 Qwen-7B）。