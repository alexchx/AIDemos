import dotenv
import re
from langchain.document_loaders import PyMuPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# References:
# https://www.zhihu.com/tardis/bd/art/640936557?source_id=1001

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


def summarize(docs, chain_type, **kwargs):
    # The splitter will only use overlap when the chunk size is longer than the chunk size limit.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )

    split_documents = text_splitter.split_documents(docs)

    chain = None
    if (chain_type == 'map_reduce'):
        chain = load_summarize_chain(llm, chain_type=chain_type, map_prompt=kwargs['map_prompt'], combine_prompt=kwargs['combine_prompt'])
    elif (chain_type == 'refine'):
        chain = load_summarize_chain(llm, chain_type=chain_type, question_prompt=kwargs['question_prompt'], refine_prompt=kwargs['refine_prompt'])
        return chain.run(split_documents)
    else:
        return None
    
    return chain.run(split_documents)


# 提取公司主要业务
def extract_com_summary(docs):
    # The splitter will only use overlap when the chunk size is longer than the chunk size limit.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    split_documents = text_splitter.split_documents(docs)

    prompt_template = "通过以下内容总结该公司经营的主要业务：\n\n\n```\n{text}\n```\n\n\n主要业务："

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type='map_reduce', map_prompt=prompt, combine_prompt=prompt)
    #chain = load_summarize_chain(llm, chain_type='refine', question_prompt=PROMPT, refine_prompt=REFINE_PROMPT)
    result = chain.run(split_documents)

    return result


# 提取公司产品及产品特征
def extract_com_products(docs):
    # The splitter will only use overlap when the chunk size is longer than the chunk size limit.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    split_documents = text_splitter.split_documents(docs)

    prompt_template = "通过以下内容总结该公司的产品及产品特征：\n\n\n```\n{text}\n```\n\n\n产品及产品特征："

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type='map_reduce', map_prompt=prompt, combine_prompt=prompt)
    #chain = load_summarize_chain(llm, chain_type='refine', question_prompt=PROMPT, refine_prompt=REFINE_PROMPT)
    result = chain.run(split_documents)

    return result


# 提取公司风险
def extract_com_risks(docs):
    # The splitter will only use overlap when the chunk size is longer than the chunk size limit.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    split_documents = text_splitter.split_documents(docs)

    prompt_template = "通过以下内容总结该公司存在哪些风险：\n\n\n```\n{text}\n```\n\n\n公司风险："

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type='map_reduce', map_prompt=prompt, combine_prompt=prompt)
    #chain = load_summarize_chain(llm, chain_type='refine', question_prompt=PROMPT, refine_prompt=REFINE_PROMPT)
    result = chain.run(split_documents)

    return result


#path = "data/恒力液压.pdf"
#path = "data/安徽众源新材料股份有限公司.pdf"
path = "data/天普股份.pdf"
#path = "data/上海韦尔半导体股份有限公司招股说明书.pdf"

print(f"\n加载文档：\n{path}")
docs = load_pdf(path)

print(f"\nTotal tokens:\n{count_tokens(docs)}")

print("\n主要业务：\n")
#result = extract_com_summary(docs)

print("\n产品及特征：\n")
#result = extract_com_products(docs)

print("\n公司风险：\n")
risk_prompt = PromptTemplate(template="通过以下内容总结该公司存在哪些风险：\n\n\n```\n{text}\n```\n\n\n公司风险：", input_variables=["text"])
result = summarize(docs, "map_reduce", map_prompt=risk_prompt, combine_prompt=risk_prompt)

print("\n竞争对手：\n")
competitors_prompt = PromptTemplate(template="通过以下内容总结该公司存在哪些竞争对手：\n\n\n```\n{text}\n```\n\n\n竞争对手：", input_variables=["text"])
result = summarize(docs, "map_reduce", map_prompt=competitors_prompt, combine_prompt=competitors_prompt)

print("\n公司客户：\n")
clients_prompt = PromptTemplate(template="通过以下内容总结该公司有哪些客户：\n\n\n```\n{text}\n```\n\n\n公司客户：", input_variables=["text"])
result = summarize(docs, "map_reduce", map_prompt=clients_prompt, combine_prompt=clients_prompt)

#3、他所在的行业位置，上、中、下游分别是什么？
#7、他的销售额、利润等财务指标

print(result)


#prompt = ChatPromptTemplate.from_template("你是一位资深的金融行业专家。请分析以下内容并找出该公司近期的供应商。请勿编造信息，答案必须使用中文。\n\n\n```\n{content}\n```")
#chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
#result = chain.run(docs)
#print(result)
