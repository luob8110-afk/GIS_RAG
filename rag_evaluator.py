import json
import chromadb
from sentence_transformers import SentenceTransformer

print("正在加载 BGE 向量模型，请稍候...")
embedding_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

print("正在连接 Chroma 数据库...")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "langchain"  # LangChain 默认集合名
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


def evaluate_retrieval(json_file_path="rag_eval_dataset.json", k_values=[1, 3, 5]):
    # 1. 加载测试集
    with open(json_file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total_queries = len(test_data)
    results = {k: 0 for k in k_values}

    print(f"\n开始评测，共计 {total_queries} 条测试数据...\n")

    # 2. 遍历测试集，开始检索
    max_k = max(k_values)
    for i, item in enumerate(test_data):
        query = item["query"]
        expected_id = item["expected_chunk_id"]

        # 将问题转为向量并查询
        query_embedding = embedding_model.encode(query).tolist()
        db_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_k
        )

        # 获取检索到的 ID 列表
        retrieved_ids = db_results['ids'][0] if db_results and 'ids' in db_results else []

        # 3. 统计命中情况
        for k in k_values:
            if expected_id in retrieved_ids[:k]:
                results[k] += 1

        if (i + 1) % 10 == 0:
            print(f"已评测 {i + 1}/{total_queries} 条...")

    # 4. 打印最终成绩单
    print("\n" + "=" * 35)
    print("🎯 RAG 系统离线检索召回率评测报告")
    print("=" * 35)
    for k in k_values:
        hit_rate = (results[k] / total_queries) * 100
        print(f"Top-{k} 命中率 (Hit Rate): {hit_rate:.2f}% ({results[k]}/{total_queries})")
    print("=" * 35)


if __name__ == "__main__":
    evaluate_retrieval()
