from sentence_transformers import SentenceTransformer
import faiss
import json

# 加载模型 & 索引 & 元数据
model = SentenceTransformer("shibing624/text2vec-base-chinese")
index = faiss.read_index("zhouyi_index.faiss")
with open("zhouyi_meta.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 查询文本
query = "童蒙是什么意思？"
query_emb = model.encode([query])

# 搜索 TopK
k = 3
D, I = index.search(query_emb, k)

print(f"\n🔍 查询: {query}")
for rank, idx in enumerate(I[0]):
    print(f"\n【Top {rank+1}】")
    print("出处：", metadata[idx]["source"])
    print("文本：", metadata[idx]["text"])
