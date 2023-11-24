import dotenv
#import langchain
from libs import qa, table_tabula, table_pdfplumber
from langchain.callbacks import get_openai_callback
from langchain.schema.language_model import BaseLanguageModel
from langchain.chat_models import ChatOpenAI


#langchain.debug = True

# References:
# https://www.zhihu.com/tardis/bd/art/640936557?source_id=1001
# https://zhuanlan.zhihu.com/p/665854051

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
#llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

def qa_pdf(path: str, llm: BaseLanguageModel):
    docs = qa.load_pdf(path)

    retriever = qa.get_vector_retriever(docs)

    #声明字符串数组
    questions = [
        "该公司的主要业务是什么",
        "该公司的产品及特征是什么",
        "该公司在行业供应链上的上、中、下游分别是什么",
        "该公司存在哪些风险",
        "该公司有哪些竞争对手",
        "该公司有哪些客户",
        "该公司的销售额、利润等财务指标"
        ]

    with get_openai_callback() as cb:
        for question in questions:
            print(f"\n\n<< {question} >>：")
            result = qa.do_qa(question, retriever, llm)
            print(result['result'])

    print("\n")

    print(cb)


def parse_tables1(path: str):
    # data/天普股份.pdf
    # [99,100]
    # [102]
    # [103]
    tables = table_tabula.read_tables(path, 'all')

    for t in tables:
        print(t)
        print("\n")

def parse_tables2(path: str):
    tables = table_pdfplumber.read_tables(path)

    table_pdfplumber.print_tables(tables)


#path = "data/宸展光电.pdf"
#path = "data/安徽众源新材料股份有限公司.pdf"
path = "data/天普股份.pdf"
#path = "data/上海韦尔半导体股份有限公司招股说明书.pdf"
#path = "data/公司.pdf"

#qa_pdf(path, llm)
#parse_tables1(path)
parse_tables2(path)
