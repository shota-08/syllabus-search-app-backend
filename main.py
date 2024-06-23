import os
import sys
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

from llm import llm_engine

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

openai_api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = openai_api_key

# 初期化
embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

# 国文学科のchroma db呼び出し
kokubun_directory = "./docs/kokubun_chroma"
kokubun_db = Chroma(collection_name="langchain_store", persist_directory=kokubun_directory, embedding_function=embedding)

# 英文学科のchroma db呼び出し
eibun_directory = "./docs/eibun_chroma"
eibun_db = Chroma(collection_name="langchain_store", persist_directory=eibun_directory, embedding_function=embedding)

# 哲学科のchroma db呼び出し
tetsugaku_directory = "./docs/tetsugaku_chroma"
tetsugaku_db = Chroma(collection_name="langchain_store", persist_directory=tetsugaku_directory, embedding_function=embedding)

# 文化史学科のchroma db呼び出し
bunkashi_directory = "./docs/bunkashi_chroma"
bunkashi_db = Chroma(collection_name="langchain_store", persist_directory=bunkashi_directory, embedding_function=embedding)

# 美学芸術学科のchroma db呼び出し
bijyutsu_directory = "./docs/bijyutsu_chroma"
bijyutsu_db = Chroma(collection_name="langchain_store", persist_directory=bijyutsu_directory, embedding_function=embedding)

class RequestData(BaseModel):
    text: str

class LLMResponse(BaseModel):
    text: str
    title_1: str
    url_1: str
    title_2: str
    url_2: str
    title_3: str
    url_3: str
    title_4: str
    url_4: str

def ask_question(query: str, db) -> str:
    docs = llm_engine.get_as_retriever_answer(query, db)
    answer = docs["result"]
    source_documents = docs["source_documents"]
    titles_urls = [(doc.metadata['title'], doc.metadata['url']) for doc in source_documents[:4]]
    return (answer, *sum(titles_urls, ()))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/kokubun")
async def run_llm(request_data: RequestData):
    answer, title_1, url_1, title_2, url_2, title_3, url_3, title_4, url_4 = ask_question(request_data.text, kokubun_db)
    return LLMResponse(text=answer,
        title_1=title_1, url_1=url_1,
        title_2=title_2, url_2=url_2,
        title_3=title_3, url_3=url_3,
        title_4=title_4, url_4=url_4
    )

@app.post("/eibun")
async def run_llm(request_data: RequestData):
    answer, title_1, url_1, title_2, url_2, title_3, url_3, title_4, url_4 = ask_question(request_data.text, eibun_db)
    return LLMResponse(text=answer,
        title_1=title_1, url_1=url_1,
        title_2=title_2, url_2=url_2,
        title_3=title_3, url_3=url_3,
        title_4=title_4, url_4=url_4
    )

@app.post("/tetsugaku")
async def run_llm(request_data: RequestData):
    answer, title_1, url_1, title_2, url_2, title_3, url_3, title_4, url_4 = ask_question(request_data.text, tetsugaku_db)
    return LLMResponse(text=answer,
        title_1=title_1, url_1=url_1,
        title_2=title_2, url_2=url_2,
        title_3=title_3, url_3=url_3,
        title_4=title_4, url_4=url_4
    )

@app.post("/bunkashi")
async def run_llm(request_data: RequestData):
    answer, title_1, url_1, title_2, url_2, title_3, url_3, title_4, url_4 = ask_question(request_data.text, bunkashi_db)
    return LLMResponse(text=answer,
        title_1=title_1, url_1=url_1,
        title_2=title_2, url_2=url_2,
        title_3=title_3, url_3=url_3,
        title_4=title_4, url_4=url_4
    )

@app.post("/bijyutsu")
async def run_llm(request_data: RequestData):
    answer, title_1, url_1, title_2, url_2, title_3, url_3, title_4, url_4 = ask_question(request_data.text, bijyutsu_db)
    return LLMResponse(text=answer,
        title_1=title_1, url_1=url_1,
        title_2=title_2, url_2=url_2,
        title_3=title_3, url_3=url_3,
        title_4=title_4, url_4=url_4
    )