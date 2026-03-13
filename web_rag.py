import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 页面基本设置 (必须放在脚本最开头)
# ==========================================
st.set_page_config(page_title="测绘/GIS 智能规范助手", page_icon="🌍", layout="centered")
st.title("🌍 测绘工程与 GIS 规范智能问答系统")
st.caption("基于大语言模型与 RAG 架构 | 支持查询重写与精准文档溯源")

# ==========================================
# 2. 配置区域 (请确保 API Key 和路径正确)
# ==========================================
QWEN_API_KEY = "sk-34de2cd3da49416abf37cc616dab3dbf"  #
DATA_FOLDER = "data/"
DB_PERSIST_DIR = "./chroma_db"


# ==========================================
# 3. 核心资源加载 (利用 Streamlit 缓存，防止每次提问都重新加载)
# ==========================================
@st.cache_resource(show_spinner="正在初始化底层大模型与知识库，请稍候...")
def init_system():
    # A. 初始化大模型 (云端)
    llm = ChatOpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0.1
    )

    # B. 初始化向量模型 (本地)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # C. 构建或加载本地知识库
    if os.path.exists(DB_PERSIST_DIR):
        vectorstore = Chroma(persist_directory=DB_PERSIST_DIR, embedding_function=embeddings)
    else:
        # 如果没有本地库，则遍历 data 文件夹读取所有 PDF
        all_docs = []
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
            st.warning(f"请在项目目录下创建 {DATA_FOLDER} 文件夹，并放入测绘/GIS相关的 PDF 规范文件！")
            st.stop()

        for filename in os.listdir(DATA_FOLDER):
            if filename.endswith(".pdf"):
                file_path = os.path.join(DATA_FOLDER, filename)
                loader = PyMuPDFLoader(file_path)
                all_docs.extend(loader.load())

        if not all_docs:
            st.warning("知识库为空！请在 data 文件夹中放入至少一个 PDF 文件。")
            st.stop()

        # 切分并存入数据库
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(all_docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_PERSIST_DIR)

    return llm, vectorstore


# 启动时执行资源初始化
llm, vectorstore = init_system()


# ==========================================
# 4. 业务逻辑：查询重写与溯源检索
# ==========================================
def chat_with_data(original_query, llm, vectorstore):
    # 【步骤一：查询重写】
    rewrite_template = """你是一个测绘与GIS领域的检索专家。请将用户的原始提问改写为更适合在专业规范文档中进行向量检索的查询语句。
    要求：提取核心专业词汇，补充全称，剥离口语化词汇。只输出改写后的查询语句。
    原始提问：{query}
    改写后的检索词："""
    rewrite_prompt = PromptTemplate.from_template(rewrite_template)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    better_query = rewrite_chain.invoke({"query": original_query})

    # 【步骤二：知识库检索】
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(better_query)

    # 【步骤三：拼接上下文与溯源信息】
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source_file = os.path.basename(doc.metadata.get('source', '未知文档'))
        page_num = doc.metadata.get('page', 0) + 1
        chunk_info = f"【来源 {i + 1}: 《{source_file}》 第 {page_num} 页】\n内容: {doc.page_content}"
        context_parts.append(chunk_info)
    context = "\n\n".join(context_parts)

    # 【步骤四：大模型最终生成】
    final_template = """你是一个严谨的测绘工程与GIS专家。请严格基于以下【参考资料】回答【用户问题】。
    要求：
    1. 如果参考资料中没有相关信息，请回答“当前知识库中暂无相关规定”，绝不捏造。
    2. 回答必须明确标出引用来源，例如：“...要求误差控制在 5mm 以内（根据《工程测量规范.pdf》第 X 页）。”

    【参考资料】：
    {context}

    【用户问题】：{query}

    【你的专业解答】："""
    final_prompt = PromptTemplate.from_template(final_template)
    final_chain = final_prompt | llm | StrOutputParser()

    response = final_chain.invoke({"context": context, "query": original_query})
    return response, better_query  # 把重写后的词也返回，方便展示给用户看


# ==========================================
# 5. Web UI 渲染与对话交互
# ==========================================
# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "你好！我是专业的 GIS 与测绘智能助手。你可以问我关于测量规范、数据处理等任何问题，我会基于底层文档给出带有出处的严谨解答。"}
    ]

# 遍历并展示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的专业问题..."):
    # 显示用户问题并存入记录
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示 AI 思考过程和最终回答
    with st.chat_message("assistant"):
        with st.spinner("思考中：正在重写意图并检索专业文档..."):
            answer, rewritten = chat_with_data(prompt, llm, vectorstore)

            # 可以在界面上用灰色小字展示大模型是怎么重写用户意图的（极其加分的亮点）
            st.caption(f"🔍 检索词优化追踪: `{rewritten}`")
            st.markdown(answer)

    # 将 AI 回答存入记录
    st.session_state.messages.append({"role": "assistant", "content": answer})