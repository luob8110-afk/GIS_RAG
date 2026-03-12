import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ==========================================
# 1. 配置区域 (修改这里)
# ==========================================
QWEN_API_KEY = "sk-34de2cd3da49416abf37cc616dab3dbf"
PDF_FILE_PATH = "data/工程测量规范.pdf"  # 确保文件名和你的实际文件一致

# ==========================================
# 2. 初始化核心组件
# ==========================================
print("正在初始化大模型与 Embedding 模型...")

# 接入通义千问大模型 (使用兼容 OpenAI 的接口格式，这是目前行业的通用做法)
llm = ChatOpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",  # qwen-turbo 速度快且免费额度高，适合开发测试
    temperature=0.1  # 降低温度，让回答更严谨，适合测绘这种严肃领域
)

# 初始化本地 Embedding 模型 (文本转向量)
# BAAI/bge-large-zh-v1.5 是非常优秀的中文开源向量模型，显存占用极小
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={'device': 'cpu'},  # 考虑到 4GB 显卡，先用 CPU 跑确保不爆显存，速度也很快
    encode_kwargs={'normalize_embeddings': True}
)


# ==========================================
# 3. 数据处理与知识库构建 (第一次运行会比较慢)
# ==========================================
def build_or_load_vector_db():
    persist_dir = "./chroma_db"

    # 如果本地已经有建好的数据库，就直接加载（节省时间）
    if os.path.exists(persist_dir):
        print("检测到本地向量库，直接加载...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("首次运行：正在解析 PDF 并构建本地知识库...")
    # 加载 PDF
    loader = PyMuPDFLoader(PDF_FILE_PATH)
    docs = loader.load()

    # 将长文档切分成 500 字的块，保留 50 字的上下文重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 存入 Chroma 数据库并持久化到本地文件夹
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("知识库构建完成！")
    return vectorstore


# ==========================================
# 4. 问答链逻辑
# ==========================================
def chat_with_data(query, vectorstore):
    print(f"\n思考中: 正在检索与 '{query}' 相关的专业资料...")

    # 从本地数据库召回最相关的 3 个文本块
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)

    # 将召回的文本块拼接到一起
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 构建 Prompt (提示词)
    template = """你是一个严谨的测绘工程与GIS专家。请严格基于以下【参考资料】回答【用户问题】。
    如果你在参考资料中找不到答案，请直接回复“当前知识库中暂无相关规定或内容”，绝不要凭空捏造。

    【参考资料】：
    {context}

    【用户问题】：{query}

    【你的专业解答】："""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    # 发送给云端大模型并获取结果
    response = chain.invoke({"context": context, "query": query})
    return response.content


# ==========================================
# 5. 主程序运行入口
# ==========================================
if __name__ == "__main__":
    db = build_or_load_vector_db()
    print("\n--- 测绘智能助手已启动 (输入 'quit' 退出) ---")

    while True:
        user_input = input("\n请提问 (例如：关于水准测量的误差限差是多少？): ")
        if user_input.lower() == 'quit':
            break

        answer = chat_with_data(user_input, db)
        print(f"\n💡 专家解答:\n{answer}")
        print("-" * 50)