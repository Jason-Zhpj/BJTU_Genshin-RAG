# ⚡ GenshinRAG: 基于 FlashRAG 的原神垂直领域知识库问答

本项目基于 [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) 框架，结合 **Qwen1.5-14B** 大模型与 **BM25/E5** 检索算法，实现了一个针对《原神》游戏知识（攻略、数值、剧情）的 RAG（检索增强生成）问答系统。

## 📂 目录结构

建议按照以下结构组织你的文件，以便于配置管理：

Plaintext

```
FlashRAG-main/
├── models/                     # [新建] 存放下载的模型权重
│   ├── Qwen1.5-14B-Chat/       # 生成模型
│   └── e5-base-v2/             # (可选) 稠密检索模型
├── indexes/                    # [新建] 存放构建好的索引文件
│   └── bm25/                   # BM25索引文件夹
├── dataset/                    # [新建] 存放数据
│   └── genshin_v2.jsonl        # 你的原神知识库文件
├── demo_zh.py                  # 问答启动脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 本说明文件
```

------

## 🛠️ 1. 环境安装

本项目建议使用 Python 3.10+ 环境。

Bash

```
# 1. 创建并激活 conda 环境
conda create -n genshin_rag python=3.10
conda activate genshin_rag

# 2. 安装 FlashRAG (开发版)
pip install flashrag-dev --pre

# 3. 安装项目其余依赖 (Streamlit, Jieba, BM25s 等)
# 确保你已上传 requirements.txt
pip install -r requirements.txt
```

> **注意**: 如果使用 NVIDIA 显卡进行推理加速，建议安装 `vllm`: `pip install vllm>=0.4.1`

------

## 📥 2. 模型下载

为了保证推理速度和稳定性，我们将模型权重下载到本地的 `./models` 目录。推荐使用 **ModelScope (魔搭社区)** 进行高速下载。

首先安装 Git LFS：

Bash

```
git lfs install
```

### 2.1 下载生成模型 (Qwen1.5-14B-Chat)

在项目根目录下执行：

Bash

```
mkdir -p models
cd models
git clone https://www.modelscope.cn/qwen/Qwen1.5-14B-Chat.git
```

### 2.2 下载检索模型 (可选)

如果你计划使用 **E5** 进行语义检索（比 BM25 更懂语义，但需要显卡资源），请下载：

Bash

```
# 在 models 目录下
git clone https://www.modelscope.cn/iic/nlp_gte_sentence-embedding_chinese-base.git
# 或者使用 huggingface 镜像下载 e5-base-v2
# git clone https://hf-mirror.com/intfloat/e5-base-v2
```

*注：如果你只使用 BM25 关键词检索，可跳过此步。*

------

## 📚 3. 数据准备

请将你的《原神》知识库整理为 `jsonl` 格式，并保存为 `dataset/genshin_v2.jsonl`。

**文件格式要求**：每一行是一个 JSON 对象，必须包含 `id` 和 `contents`。

代码段

```
{"id": "1", "contents": "标题：胡桃攻略\n内容：胡桃是火属性长柄武器角色，推荐圣遗物为魔女四件套...", "title": "胡桃攻略"}
{"id": "2", "contents": "标题：那维莱特\n内容：那维莱特主要依靠重击输出，核心命座是1命...", "title": "那维莱特"}
```

------

## 🏗️ 4. 构建索引 (Index Construction)

在运行问答之前，必须先对语料库建立索引。

### 方案 A: 构建 BM25 稀疏索引 (推荐，速度快)

本项目默认使用 `bm25s` 库，无需显卡即可快速构建。

Bash

```
# 回到项目根目录执行
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path dataset/genshin_v2.jsonl \
    --bm25_backend bm25s \
    --save_dir indexes/bm25
```

执行完成后，检查 `indexes/bm25` 目录下是否有生成的 `.json` 或 `.parquet` 文件。

### 方案 B: 构建 E5 稠密索引 (语义效果更好)

需要 GPU 支持。

Bash

```
CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path models/e5-base-v2 \
    --corpus_path dataset/genshin_v2.jsonl \
    --save_dir indexes/e5 \
    --use_fp16 \
    --max_length 512 \
    --batch_size 64 \
    --pooling_method mean \
    --faiss_type Flat
```

------

## ⚙️ 5. 修改配置与运行

### 5.1 修改路径配置

打开 `demo_zh.py`，找到 `config_dict` 部分。为了防止路径错误，建议使用 `os.path` 自动获取绝对路径：

Python

```
import os

# 获取当前项目根目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config_dict = {
    "save_note": "demo",
    "generator_model": "qwen-14B",
    
    # 【关键】指向刚才下载的模型路径
    "model2path": {
        "qwen-14B": os.path.join(BASE_DIR, "models/Qwen1.5-14B-Chat"),
        # 如果使用 e5，取消注释并修改路径
        # "e5": os.path.join(BASE_DIR, "models/e5-base-v2"),
    },
    
    # 指向你的 jsonl 语料
    "corpus_path": os.path.join(BASE_DIR, "dataset/genshin_v2.jsonl"),
    
    # 检索配置 (默认 BM25)
    "retrieval_method": "bm25", 
    "index_path": os.path.join(BASE_DIR, "indexes/bm25"), 
    "bm25_backend": "bm25s",
}
```

### 5.2 启动 WebUI

在终端运行以下命令：

Bash

```
streamlit run demo_zh.py
```

### 5.3 使用说明

1. 浏览器将自动打开 (默认地址 `http://localhost:8501`)。
2. 在侧边栏调整参数：
   - **Temperature**: 0.1-0.5 (越低回答越严谨，适合知识问答)。
   - **Top-K**: 3-5 (每次参考的文档数量)。
3. 输入问题，例如：“**芙宁娜的元素爆发机制是什么？**”
4. 点击 **Generate Responses**，系统将展示：
   - 检索到的参考文档 (References)
   - RAG 回答 (基于文档生成的答案)
   - 非 RAG 回答 (模型裸跑的答案，用于对比)

------

## ❓ 常见问题 FAQ

**Q: 报错 `Index not found`?** A: 请检查 `config_dict` 中的 `index_path` 是否正确指向了包含索引文件的**文件夹**（对于 BM25）或**具体文件**（对于 Faiss/E5）。

**Q: 显存不足 (OOM)?** A: Qwen-14B 大约需要 28GB+ 显存。显存不够可尝试：

1. 在 `config_dict` 添加量化参数（需安装 bitsandbytes）： `"generator_load_kwargs": {"load_in_8bit": True}`
2. 更换更小的模型，如 `Qwen1.5-7B-Chat`。

**Q: BM25 检索结果不准确？** A: 可以在 `demo_zh.py` 的 `preprocess_query` 函数中优化分词逻辑，或者增加 `genshin_v2.jsonl` 中的数据丰富度。