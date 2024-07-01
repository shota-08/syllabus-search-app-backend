import pandas as pd
from langchain_openai import OpenAIEmbeddings
import chromadb
import chromadb.config
import uuid

from llm_engine import get_llm_summary, get_llm_keyword

from dotenv import load_dotenv
load_dotenv()

# 初期化
embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

# 読み込み用データ
df = pd.read_csv("./csv/kokubun/kokubun_授業計画入り.csv")
titles = df["title"].tolist()
urls = df["url"].tolist()
teachers = df["teacher"].tolist()
ids = df["id"].tolist()
texts = df["text"].tolist()
summary_texts = [get_llm_summary(text) for text in texts]
contents = df["content"].tolist()
summary_contents = [get_llm_keyword(content) for content in contents]
df["summary_texts"] = summary_texts
df["summary_contents"] = summary_contents

# embedding用カラム作成
df["anotation"] = df.apply(lambda row: f"# 講義名\n{row['title']}\n\n# 概要\n{row['summary_texts']}\n\n# 授業題材\n{row['summary_contents']}", axis=1)
anotations = df["anotation"].tolist()
embeddings = embedding.embed_documents(anotations)

# chromadb
persist_directory = "./docs/kokubun_chroma"
collection_name="langchain_store"

client = chromadb.PersistentClient(path=persist_directory)
collection = client.create_collection(name=collection_name)

collection.add(
    documents = anotations,
    embeddings = embeddings,
    metadatas = [{"title": s, "url": l, "teacher": m, "id": n } for s, l, m, n in zip(titles, urls, teachers, ids)],
    ids=[str(uuid.uuid1()) for _ in anotations]
)

# 確認用
# df.to_csv("./csv/kokubun_llm.csv", index=False)