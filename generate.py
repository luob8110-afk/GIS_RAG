import json
import random
import chromadb
from openai import OpenAI

# ================= 1. 配置区域 =================
API_KEY = "miyao"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# Chroma 数据库路径和集合名称
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "langchain"

# ===============================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_test_dataset(sample_size=100, output_file="rag_eval_dataset.json"):
    print(f"正在连接本地 Chroma 数据库: {CHROMA_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # 1. 一次性拉取数据库中所有的 Chunk 文本和对应的 ID
    print("正在拉取文档数据...")
    all_data = collection.get(include=['documents'])
    all_ids = all_data['ids']
    all_docs = all_data['documents']

    total_docs = len(all_ids)
    print(f"数据库中共找到 {total_docs} 个文本片段(Chunk)。")

    # 2. 随机抽取指定数量的样本
    combined = list(zip(all_ids, all_docs))
    random.shuffle(combined)
    sampled_data = combined[:min(sample_size, total_docs)]

    test_dataset = []

    print(f"\n开始调用大模型反向生成 {len(sampled_data)} 条测试QA，请稍候...")

    for i, (chunk_id, chunk_text) in enumerate(sampled_data):
        # 过滤掉字数太少、没法提问的垃圾数据
        if len(chunk_text.strip()) < 30:
            continue

        # 3. 核心 Prompt 工程：让大模型扮演提问者
        prompt = f"""你是一个测绘工程与GIS领域的业务人员。
请阅读以下内部文档片段，并基于这段内容，提出 **1个** 具体的业务问题。
要求：
1. 提问的内容必须能在这个文档片段中找到答案。
2. 问题要稍微口语化一些，符合真实的日常提问习惯。
3. 请直接输出问题本身，不要输出任何解释、寒暄或换行符。

文档片段：
{chunk_text}
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个严谨的测绘数据语料处理助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7  # 给一点温度值，让生成的问题更多样化
            )
            generated_query = response.choices[0].message.content.strip()

            # 4. 组装数据并保存
            test_dataset.append({
                "query": generated_query,
                "expected_chunk_id": chunk_id
            })
            print(f"进度 [{i + 1}/{len(sampled_data)}] | 生成问题: {generated_query}")

        except Exception as e:
            print(f"进度 [{i + 1}/{len(sampled_data)}] | API 调用失败: {e}")

    # 5. 导出为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 自动评测集生成完毕！已成功保存至 {output_file}")


if __name__ == "__main__":
    generate_test_dataset(sample_size=100)
