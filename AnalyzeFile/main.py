import dotenv
import re
from langchain.document_loaders import PyPDFLoader

from pprint import pprint

# References:
# https://www.zhihu.com/tardis/bd/art/640936557?source_id=1001

dotenv.load_dotenv()

path = "data/恒力液压.pdf"

loader1 = PyPDFLoader(path)
pages1 = loader1.load()
#pprint(pages1[32:37])

slice = pages1[32:37]
exp = "^[^\\n]+招股说明书  \\n \\d+-\\d+-\\d+ "

content = "\n".join([re.sub(exp, "", page.page_content) for page in slice])
pprint(content)

#from langchain.document_loaders import PyMuPDFLoader
#loader2 = PyMuPDFLoader(path)
#pages2 = loader2.load()
#print(pages2[0])

#from langchain.document_loaders import  UnstructuredPDFLoader
#loader3 = UnstructuredPDFLoader(path)
#pages3 = loader3.load()
#print(pages3[0])

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_documents = text_splitter.create_documents([content])
print(f'documents: {len(split_documents)}')

prompt_template = """写出以下内容的简明摘要：


"{text}"


简明摘要："""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, streaming=False)
chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=True, map_prompt=prompt, combine_prompt=prompt)
chain.run(split_documents)
