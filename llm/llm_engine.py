import os
import openai

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()  # read local .env file

openai_api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = openai_api_key

# LLMモデルの設定
llm_3 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_4 = ChatOpenAI(model="gpt-4o", temperature=0)

def get_as_retriever_answer(query, db):
    prompt_template = """
    # 命令
    あなたは文学部の進路相談のアドバイザーです。
    学生の問い合わせに親切に回答してください。
    そして講義名と概要を参照しながら、おすすめの講義を最大で3つまで紹介してください。

    # 制約条件:
    ・講義名と概要を簡潔に説明すること
    ・学生の興味をそそるために、おすすめした理由やジャンルや年代や題材や具体的なキーワードを概要の説明に含めること
    ・講座概要に書いていないことは伝えてはいけない
    ・単位取得の情報、欠席の情報、履修条件の情報は説明に含めてはいけない
    ・300文字以内で簡潔に答える

    # 参考情報:
    {context}

    # 学生の問い合わせ:
    {question}

    # 回答例:
    (ここは自由に文章を書きなさい。)
    1. **(おすすめの授業名)**
        - **概要**:
        - **おすすめの理由**:
    2. **(おすすめの授業名)**
        - **概要**:
        - **おすすめの理由**:
    3. **(おすすめの授業名)**
        - **概要**:
        - **おすすめの理由**:
    """
    prompt_qa = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt_qa}
    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=llm_4,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )
    result = qa(query)
    return result