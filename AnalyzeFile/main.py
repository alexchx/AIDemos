import dotenv
import re
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

# References:
# https://www.zhihu.com/tardis/bd/art/640936557?source_id=1001
# https://zhuanlan.zhihu.com/p/665854051

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
#llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

def load_pdf(path):
    loader = PyMuPDFLoader(path)
    docs = loader.load()

    exp = "^([^\\n]+)公司([\\n ]*)([^\\n]*)招股说明书([^\\n]*)([\\n ]+)([\\d\\- ]+)([\\n ]+)"

    # 去除页眉页脚
    for i in range(len(docs)):
        docs[i].page_content = re.sub(exp, "", docs[i].page_content)

    return docs


def count_tokens(docs):
    num_tokens = 0
    for doc in docs:
        num_tokens += llm.get_num_tokens(doc.page_content)
    return num_tokens


def get_vector_retriever(docs):
    # The splitter will only use overlap when the chunk size is longer than the chunk size limit.
    # https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(deployment="lqembedding")

    docsearch = Chroma.from_documents(split_docs, embeddings)

    return docsearch.as_retriever()


def do_qa(question, retriever):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    result = qa({"query": question})

    return result


path = "data/恒力液压.pdf"
#path = "data/安徽众源新材料股份有限公司.pdf"
#path = "data/天普股份.pdf"
#path = "data/上海韦尔半导体股份有限公司招股说明书.pdf"

docs = load_pdf(path)
tokens = count_tokens(docs)
print(f"{path}\n{tokens} tokens")

retriever = get_vector_retriever(docs)

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

for question in questions:
    print(f"\n\n<< {question} >>：")
    result = do_qa(question, retriever)
    print(result['result'])
