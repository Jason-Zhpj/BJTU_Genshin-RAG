import json
import jieba
from flashrag.config import Config
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate
from tqdm import tqdm  # 用于显示进度条，如果没有安装可以去掉

# --- 配置部分 ---
STOPWORDS = {"是谁", "是", "谁", "的", "呢", "吗", "了", "关于", "介绍", "一下"}

config_dict = {
    "save_note": "demo",
    "generator_model": "qwen-14B",
    "model2path": {"qwen-14B": "/data/wzy/zy/LLM-RAG/models/Qwen1.5-14B-Chat"},
    "corpus_path": "/data/wzy/zy/LLM-RAG/models/genshin_v2.jsonl",
    "retrieval_method": "bm25",
    "index_path": "/data/wzy/zy/LLM-RAG/models/indexes/bm25",
    "bm25_backend": "bm25s",
}

# 待测试的10个问题
QUESTIONS = [
    "有什么角色适合充当肉盾聚怪？",
    "七七一般在团队充当什么角色？",
    "北斗是什么？",
    "钟离是谁？",
    "外号为女士的人被谁斩杀了？",
    "史莱姆是什么？",
    "蒙德是什么？",
    "胡桃是谁？",
    "原石是什么？",
    "甘雨是谁？"
]

# 生成参数
TEMPERATURE = 0.5
TOPK = 5
MAX_NEW_TOKENS = 1024
OUTPUT_FILE = "rag_benchmark_results.json"

def preprocess_query(query):
    # 使用 jieba 进行精确分词
    words = jieba.lcut(query)
    # 过滤掉停用词
    filtered_words = [w for w in words if w not in STOPWORDS and len(w.strip()) > 0]
    return " ".join(filtered_words)

def main():
    print("正在加载配置和模型...")
    config = Config("my_config.yaml", config_dict=config_dict)
    
    # 加载生成器和检索器
    generator = get_generator(config)
    retriever = get_retriever(config)

    # 定义 Prompt 模板
    system_prompt_rag = (
        "你是一位精通《原神》（Genshin Impact）的游戏知识库助手。你的任务是根据下方的【参考文档】回答用户关于游戏机制、角色攻略、剧情或数值的问题。\n"
        "请严格遵守以下规则：\n"
        "1. **绝对忠实**：你的所有回答必须严格基于【参考文档】中的信息。严禁使用你原本的训练知识来回答（因为游戏版本更新频繁，你的旧知识可能是错误的）。\n"
        "2. **拒绝幻觉**：如果【参考文档】中没有包含回答问题所需的信息，请直接诚实地回答：“抱歉，当前的资料库中没有找到相关信息。”，不要编造数值或机制。\n"
        "3. **结构清晰**：使用Markdown格式（如列表、粗体）来组织答案，使攻略看起来直观易读。\n"
        "4. **语气友好**：保持乐于助人的态度，称呼用户为“旅行者”。\n\n"
        "【参考文档】：\n"
        "{reference}"
    )
    
    system_prompt_no_rag = (
        "你是一个友好的人工智能助手。请对用户的输出做出高质量的响应，生成类似于人类的内容，并尽量遵循输入中的指令。\n"
    )
    
    base_user_prompt = "{question}"

    prompt_template_rag = PromptTemplate(config, system_prompt=system_prompt_rag, user_prompt=base_user_prompt)
    prompt_template_no_rag = PromptTemplate(config, system_prompt=system_prompt_no_rag, user_prompt=base_user_prompt)

    results = []

    print(f"开始批量处理 {len(QUESTIONS)} 个问题...")
    
    # 遍历问题
    # 如果没安装 tqdm，可以将 range 里的 tqdm(QUESTIONS) 改为 QUESTIONS
    for i, query in enumerate(tqdm(QUESTIONS, desc="Processing")):
        
        # 1. 检索 (Retrieval)
        clean_query = preprocess_query(query)
        retrieved_docs = retriever.search(clean_query, num=TOPK)
        
        # 提取检索到的文本供参考（可选，方便调试）
        retrieved_contents = [doc["contents"] for doc in retrieved_docs]

        # 2. 生成 - With RAG
        input_prompt_with_rag = prompt_template_rag.get_string(question=query, retrieval_result=retrieved_docs)
        response_with_rag = generator.generate(
            input_prompt_with_rag, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
        )[0]

        # 3. 生成 - Without RAG
        input_prompt_without_rag = prompt_template_no_rag.get_string(question=query)
        response_without_rag = generator.generate(
            input_prompt_without_rag, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
        )[0]

        # 4. 整理结果
        result_item = {
            "id": i + 1,
            "question": query,
            "response_with_rag": response_with_rag,
            "response_without_rag": response_without_rag,
            "retrieved_docs": retrieved_contents # 同时也保存检索到的文档内容，方便分析
        }
        results.append(result_item)

    # 5. 保存到 JSON 文件
    print(f"正在保存结果到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("完成！")

if __name__ == '__main__':
    main()