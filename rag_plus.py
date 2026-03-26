import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ==========================================
# 1. 配置区域 (修改这里)
# ==========================================
QWEN_API_KEY = "密钥"
# PDF_FILE_PATH = "data/工程测量规范.pdf"  # 确保文件名和你的实际文件一致

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
    model_kwargs={'device': 'cuda'},  # 切换到gpu
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

    # 1. 遍历 data 文件夹，加载所有 PDF
    all_docs = []
    data_folder = "data/"
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            print(f"正在读取: {filename} ...")
            loader = PyMuPDFLoader(file_path)
            all_docs.extend(loader.load())  # 将所有文档的页面追加到大列表中

    if not all_docs:
        raise ValueError("在 data 文件夹中没有找到 PDF 文件！")

    # 2. 对合并后的所有文档进行统一切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)

    # 3. 存入 Chroma 数据库并持久化
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )


# ==========================================
# 4. 问答链逻辑
# ==========================================
def rewrite_query(original_query):
    """
    [优化4：查询重写] 将用户的口语化提问，转化为专业且富含关键词的检索语句
    """
    print("\n🔄 [处理中] 正在通过大模型重写你的查询，提取核心测绘/GIS词汇...")

    rewrite_template = """你是一个测绘与GIS领域的检索专家。
    请将用户的原始提问改写为更适合在专业规范文档中进行向量检索的查询语句。
    要求：
    1. 提取核心专业词汇，补充可能的全称或学名（例如把 RTK 补充为 实时动态差分法）。
    2. 剥离口语化的无用词汇（如“我想知道”、“告诉我”）。
    3. 必须只输出改写后的查询语句，不要输出任何解释。

    原始提问：{query}
    改写后的检索词："""

    prompt = PromptTemplate.from_template(rewrite_template)
    # StrOutputParser 可以直接把大模型的复杂输出剥离成纯文本字符串
    chain = prompt | llm | StrOutputParser()

    rewritten_query = chain.invoke({"query": original_query})
    print(f"🎯 [重写结果] => {rewritten_query.strip()}")
    return rewritten_query.strip()


def chat_with_data(original_query, vectorstore):
    """
    带溯源能力的问答主函数
    """
    # 第一步：调用查询重写 (Opt 4)
    better_query = rewrite_query(original_query)

    print("🔎 [检索中] 正在本地知识库中寻找最匹配的规范条款...")

    # 第二步：使用重写后的更好词汇去检索 Top-3 文本块
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(better_query)

    # 第三步：组装带有 Metadata (来源页码) 的上下文 (Opt 2)
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        # 提取文件名（去掉路径，只留文件名）
        source_file = os.path.basename(doc.metadata.get('source', '未知文档'))
        # 提取页码 (LangChain 默认页码从 0 开始，所以为了人类阅读习惯加 1)
        page_num = doc.metadata.get('page', 0) + 1

        # 将来源信息和文本块拼接在一起
        chunk_info = f"【来源 {i + 1}: 《{source_file}》 第 {page_num} 页】\n内容: {doc.page_content}"
        context_parts.append(chunk_info)

    context = "\n\n".join(context_parts)

    # 第四步：带有严格格式要求的 Prompt (Opt 2)
    template = """你是一个严谨的测绘工程与GIS专家。请严格基于以下【参考资料】回答【用户问题】。

    【回答要求】：
    1. 如果参考资料中没有相关信息，请直接回答“当前知识库中暂无相关规定”，绝不要凭空捏造。
    2. 你的回答必须明确标出引用来源。格式要求：在陈述完一个观点后，在括号内注明出处，如：“...要求误差控制在 5mm 以内（根据《工程测量规范.pdf》第 X 页）。”

    【参考资料】：
    {context}

    【用户问题】：{query}

    【你的专业解答】："""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 注意：最终回答时，依然传入用户原始的口语化提问，这样回答显得更自然
    response = chain.invoke({"context": context, "query": original_query})
    return response

# ==========================================
# 5. 主程序运行入口
# ==========================================
if __name__ == "__main__":
    db = build_or_load_vector_db()
    print("\n" + "=" * 50)
    print("🚀 高阶测绘智能助手已启动 (包含查询重写 & 来源溯源)")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n🧑‍💻 你的问题: ")
        if user_input.lower() == 'quit':
            break

        answer = chat_with_data(user_input, db)
        print(f"\n💡 专家解答:\n{answer}")
        print("-" * 50)
