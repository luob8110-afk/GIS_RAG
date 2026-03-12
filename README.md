### GIS_RAG
# 基于 LLM 应用开发框架构建检索增强生成（RAG）系统，解决大模型在特定垂直领域（如测绘专业知识）的幻觉问题

## 技术栈：
LangChain，PyMuPDF + RecursiveCharacterTextSplitter，BAAI/bge-large-zh-v1.5，ChromaDB，Qwen API

# 第一步：搭建项目结构与环境
首先创建文件夹GIS_RAG，然后在根目录创建data文件夹存储数据集，并在根目录创建运行代码

之后打开终端配置环境，执行下面的命令

```bash
pip install langchain langchain-community langchain-openai langchain-huggingface chromadb pymupdf sentence-transformers
```

# 第二步：
收集专业数据，pdf版文档，例如测绘工程等专业规范

# 第三步：
编辑核心代码，已经附带了rag_app.py文件里面。

# 第四步
运行即可
```bash
python rag_app.py
```
<img src="images/first_complete.png" width="500">
<img src="images/fst_question1.png" width="500">

# 操作遇到的问题:
在配置环境时出现了缺少微软的 C++ 编译器，一直显示报错，解决办法是#安装了Download Build Tools。
