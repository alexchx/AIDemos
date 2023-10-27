import dotenv
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

dotenv.load_dotenv()

loader = UnstructuredFileLoader('data/content.txt')
document = loader.load()

# https://githubhelp.com/hwchase17/langchain/issues/8142
# The splitter will only use overlap when the chunk size is longer than the chunk size limit.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_documents = text_splitter.split_documents(document)
print(f'documents: {len(split_documents)}')

prompt_template = """写出以下内容的简明摘要：


"{text}"


简明摘要："""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, streaming=False)
chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=True, map_prompt=prompt, combine_prompt=prompt)
chain.run(split_documents)
