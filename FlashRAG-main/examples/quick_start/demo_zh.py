import streamlit as st
from flashrag.config import Config
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate

import jieba

STOPWORDS = {"是谁", "是", "谁", "的", "呢", "吗", "了", "关于", "介绍", "一下"}

config_dict = {
    "save_note": "demo",
    "generator_model": "qwen-14B",
    "model2path": {"qwen-14B": "/data/wzy/zy/LLM-RAG/models/Qwen1.5-14B-Chat"},
    "corpus_path": "/data/wzy/zy/LLM-RAG/models/genshin_v2.jsonl",
    "retrieval_method":"bm25",
    "index_path":"/data/wzy/zy/LLM-RAG/models/indexes/bm25",
    "bm25_backend":"bm25s",
}
    #"retrieval_method": "e5",
    # "model2path": {"e5": "/data/wzy/zy/LLM-RAG/models/e5-base-v2", "qwen-14B": "/data/wzy/zy/LLM-RAG/models/Qwen1.5-14B-Chat"},
    #"index_path": "/data/wzy/zy/LLM-RAG/models/indexes/genshin/e5_Flat.index",

def preprocess_query(query):
    # 使用 jieba 进行精确分词
    words = jieba.lcut(query)
    # 过滤掉停用词
    filtered_words = [w for w in words if w not in STOPWORDS and len(w.strip()) > 0]
    # 重新拼接成关键词字符串，bm25s 通常接受空格分隔的字符串
    return " ".join(filtered_words)

@st.cache_resource
def load_retriever(_config):
    return get_retriever(_config)


@st.cache_resource
def load_generator(_config):
    return get_generator(_config)

if __name__ == '__main__':

    ## 导入 debugpy 库，debugpy 是一个用于在 Python 中进行调试的库，通常与 Visual Studio Code 配合使用
    #import debugpy
    #try:
    #    # 调用 debugpy 的 listen 方法，使调试器开始监听指定的主机和端口。在这里，监听的主机是 'localhost'，端口是 9501。默认情况下，VS Code 调试配置会使用 5678 端口，但这里使用了 9501。
    #    debugpy.listen(("localhost", 9501))
    #     # 输出信息，提示用户调试器正在等待附加连接
    #    print("Waiting for debugger attach")
    #    # 等待调试器（例如 VS Code）连接到当前 Python 进程。程序会在这一行暂停，直到调试器附加进来。
    #    debugpy.wait_for_client()
    ## 捕获所有异常，若有异常发生，进入 except 块
    #except Exception as e:
    #    # 如果发生异常，则什么也不做，直接跳过
    #    pass

    custom_theme = {
        "primaryColor": "#ff6347",
        "backgroundColor": "#f0f0f0",
        "secondaryBackgroundColor": "#d3d3d3",
        "textColor": "#121212",
        "font": "sans serif",
    }
    st.set_page_config(page_title="FlashRAG Demo", page_icon="⚡")


    st.sidebar.title("Configuration")
    temperature = st.sidebar.slider("Temperature:", 0.01, 1.0, 0.5)
    topk = st.sidebar.slider("Number of retrieved documents:", 1, 10, 5)
    max_new_tokens = st.sidebar.slider("Max generation tokens:", 1, 2048, 1024)# max length of the output，具体参数含义为生成的最大token数


    st.title("⚡GenshinRAG-BJTU")
    st.write("基于Qwen-14B的的垂直领域RAG系统——以原神为例")


    query = st.text_area("Enter your prompt:")

    config = Config("my_config.yaml", config_dict=config_dict)
    generator = load_generator(config)
    retriever = load_retriever(config)

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
        "你是一个友好的人工智能助手。" "请对用户的输出做出高质量的响应，生成类似于人类的内容，并尽量遵循输入中的指令。\n"
    )
    base_user_prompt = "{question}"

    prompt_template_rag = PromptTemplate(config, system_prompt=system_prompt_rag, user_prompt=base_user_prompt)
    prompt_template_no_rag = PromptTemplate(config, system_prompt=system_prompt_no_rag, user_prompt=base_user_prompt)


    if st.button("Generate Responses"):
        with st.spinner("Retrieving and Generating..."):

            clean_query=preprocess_query(query)

            retrieved_docs = retriever.search(clean_query, num=topk)

            st.subheader("References", divider="gray")
            for i, doc in enumerate(retrieved_docs):
                doc_title = doc.get("title", "No Title")
                doc_text = "\n".join(doc["contents"].split("\n")[1:])
                expander = st.expander(f"**[{i+1}]: {doc_title}**", expanded=False)
                with expander:
                    st.markdown(doc_text, unsafe_allow_html=True)

            st.subheader("Generated Responses:", divider="gray")

            input_prompt_with_rag = prompt_template_rag.get_string(question=query, retrieval_result=retrieved_docs)
            response_with_rag = generator.generate(
                input_prompt_with_rag, temperature=temperature, max_new_tokens=max_new_tokens
            )[0]
            st.subheader("Response with RAG:")
            st.write(response_with_rag)
            input_prompt_without_rag = prompt_template_no_rag.get_string(question=query)
            response_without_rag = generator.generate(
                input_prompt_without_rag, temperature=temperature, max_new_tokens=max_new_tokens
            )[0]
            st.subheader("Response without RAG:")
            st.markdown(response_without_rag)
