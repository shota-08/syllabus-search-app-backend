from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()  # read local .env file

# LLMモデルの設定
llm_3 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_4 = ChatOpenAI(model="gpt-4o", temperature=0)

# ○○学科の部分は適宜変更
def get_llm_summary(text):
    prompt_template = """
    # 命令
    あなたは、日本の大学の文学部国文学科の履修登録のアドバイザーです。
    履修登録を検討している大学生に対して、この講義で学べる題材は何か、200文字以内で簡潔に説明してください。
    一度大きく深呼吸をしてステップバイステップで考えてください。

    # 制約条件:
    ・200文字以内で簡潔に答える
    ・講義で扱うジャンルや年代や題材のキーワードは、可能な限り説明に含めること
    ・単位取得の情報、欠席の情報、履修条件の情報は説明に含めてはいけない
    ・複数の要素がある場合、文化史における重要度の高い要素から優先して説明すること

    # 講義内容:
    {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm_4
    answer = chain.invoke({"text": text})
    return answer.content

# "文学部○○学科"と"○○に関連するキーワード"の部分は適宜変更
def get_llm_keyword(text):
    prompt_template = """
    # 命令
    あなたは、日本の大学の文学部国文学科の履修アシスタントです。
    以下の授業計画を参照し、国文学に関連するキーワードや作品名を抽出しなさい。
    もし内容がない場合、何も生成せずに「シラバス参照」と返しなさい。

    # 授業計画:
    {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm_4
    answer = chain.invoke({"text": text})
    return answer.content
