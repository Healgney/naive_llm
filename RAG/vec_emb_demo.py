from sentence_transformers import SentenceTransformer
import faiss
import os
import json

# Step 1: 选择嵌入模型（中文专用）
model_name = "shibing624/text2vec-base-chinese"
embed_model = SentenceTransformer(model_name)

# Step 2: 加载文档（支持多个 .txt 文件）
def load_documents(folder):
    docs = []
    metadata = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                content = f.read()
                # 切分为段落或句子
                chunks = [line.strip() for line in content.split("\n") if line.strip()]
                for i, chunk in enumerate(chunks):
                    docs.append(chunk)
                    metadata.append({
                        "source": fname,
                        "line": i,
                        "text": chunk
                    })
    return docs, metadata

docs, meta = load_documents("/tmp/pycharm_project_105/data/raw")

# Step 3: 文本向量化
embeddings = embed_model.encode(docs, show_progress_bar=True)

# Step 4: 构建 FAISS 索引
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)  # 使用 L2 距离
index.add(embeddings)

# Step 5: 保存 index + metadata
faiss.write_index(index, "zhouyi_index.faiss")
with open("zhouyi_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ 索引构建完成，文档数：{len(docs)}，向量维度：{dim}")
