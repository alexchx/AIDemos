import dotenv
from meutils.pipe import *
from chatllm.llmchain.applications import ChatOCR

dotenv.load_dotenv()

llm = ChatOCR()
file_path = "data/invoice.jpg"
llm.display(file_path, 700)
llm.chat('识别编号,公司名称,开票日期,开票人,收款人,复核人,金额', file_path=file_path) | xprint
