#import langchain
import re
from langchain.chains import RetrievalQA
from langchain.schema.language_model import BaseLanguageModel
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from typing import Iterable
from langchain.docstore.document import Document

#langchain.debug = True

# References:
# https://www.zhihu.com/tardis/bd/art/640936557?source_id=1001
# https://zhuanlan.zhihu.com/p/665854051

def load_pdf(path: str):
    loader = PyMuPDFLoader(path)
    docs = loader.load()

    exp = "^([^\n]+)公司([\n ]*)([^\n]*)招股说明书([^\n]*)([\n ]+)([\d\- ]+)([\n ]+)"

    # 去除页眉页脚
    for i in range(len(docs)):
        docs[i].page_content = re.sub(exp, "", docs[i].page_content)

    return docs


def count_tokens(docs: Iterable[Document], llm: BaseLanguageModel):
    num_tokens = 0
    for doc in docs:
        num_tokens += llm.get_num_tokens(doc.page_content)
    return num_tokens


def get_vector_retriever(docs: Iterable[Document]):
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


def do_qa(question: str, retriever, llm: BaseLanguageModel):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    result = qa({"query": question})

    return result
