from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

from llm import llm_engine

# 初期化
embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

# chroma db呼び出し
persist_directory = "./docs/chroma"
db = Chroma(collection_name="langchain_store", persist_directory=persist_directory, embedding_function=embedding)

class RequestData(BaseModel):
    text: str

class LLMResponse(BaseModel):
    text: str
    title: str
    url: str
    content: str

def ask_question(query: str) -> str:
    docs = db.similarity_search(query, k=2)
    answer_meta = docs[0].metadata
    answer_title = answer_meta['title']
    answer_url = answer_meta['url']
    answer_content = docs[0].page_content
    answer = llm_engine.get_llm_answer(query, answer_title, answer_content)
    return answer, answer_title, answer_url, answer_content

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/send")
async def run_llm(request_data: RequestData):
    answer, title, url, content = ask_question(request_data.text)
    return LLMResponse(text=answer, title=title, url=url, content=content)