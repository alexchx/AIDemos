#import langchain
import os
import re
import cn2an
from langchain.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from typing import Optional, List, Self
from colorama import Fore

#langchain.debug = True

showTrace = False

# References:
# https://www.zhihu.com/tardis/bd/art/640936557?source_id=1001
# https://zhuanlan.zhihu.com/p/665854051

nums_zh = "一二三四五六七八九十零"
exp_nums_zh = f"[{nums_zh}]+"
exp_no_linebreak = "[^\n]+"
exp_line_prefix = "((?<=\n) *|^)"
exp_line_postfix = "((?= *\n)|$)"
exp_header_types = [
    f"{exp_line_prefix}(本次发行概况|发行概况|发行人声明|释义|目[\n ]*录|声明及承诺|重大事项提示|重要说明){exp_line_postfix}",
    f"{exp_line_prefix}第({exp_nums_zh})节\n*{exp_no_linebreak}",   # 第一节
    f"{exp_line_prefix}({exp_nums_zh})、{exp_no_linebreak}",     # 一、
    f"{exp_line_prefix}（({exp_nums_zh})）{exp_no_linebreak}",   # （一）
    f"{exp_line_prefix}(\d+)、{exp_no_linebreak}",               # 1、
    f"{exp_line_prefix}（(\d+)）{exp_no_linebreak}"              # （1）
]
exp_header = f"({'|'.join(exp_header_types)})"


class OutlineHeader:
    label: str
    parents: list[Self]
    children: list[Self]

    def __init__(self, label: str):
        self.label = label
        self.parents = []
        self.children = []

    def add_child(self, child: Self) -> bool:
        prevOrder = header_info(self.children[len(self.children) - 1].label)['order'] if len(self.children) > 0 else 0
        curOrder = header_info(child.label)['order']

        if curOrder != prevOrder + 1 and (curOrder != 0 or prevOrder != 0):
            # header order not match sibling or not start from 1
            if showTrace:
                print(f"Ignored header as invalid order {curOrder}: {child.label}")
            return False

        child.parents = self.parents.copy()
        child.parents.append(self)

        self.children.append(child)

        return True

    def add_sibling(self, sibling: Self) -> bool:
        prevOrder = header_info(self.label)['order']
        curOrder = header_info(sibling.label)['order']

        if curOrder != prevOrder + 1 and (curOrder != 0 and prevOrder != 0):
            # header order not match sibling or not start from 1
            if showTrace:
                print(f"Ignored header as invalid order {curOrder}: {sibling.label}",)
            return False

        sibling.parents = self.parents
            
        self.parents[len(self.parents) - 1].children.append(sibling)

        return True


def load_pdf(path: str):
    loader = PyMuPDFLoader(path)
    docs = loader.load()

    exp = "^([^\n]+)公司([\n ]*)([^\n]*)招股说明书([^\n]*)([\n ]+)([\d\- ]+)([\n ]+)"

    # 去除页眉页脚
    for i in range(len(docs)):
        docs[i].page_content = re.sub(exp, "", docs[i].page_content)

    return docs


def extract_headers(pages: List[Document]):
    regex = re.compile(exp_header, flags=re.MULTILINE)

    # TODO：先去除表格内容后再提取标题，因表格中可能会含有很多异常的标题信息

    headers = []
    for page in pages:
        matches = regex.findall(page.page_content)
        for m in matches:
            headers.append(m[0])
    
    # 去除标题中的换行符、以及首尾空格
    regex = re.compile("\n+")
    headers = [regex.sub(" ", h.strip()) for h in headers]

    # 去除目录内容
    catalogIndex = -1
    for i in range(len(headers)):
        if re.search("目[\n ]*录", headers[i]):
            catalogIndex = i
            break
    lastCatalogItemIndex = -1
    if catalogIndex != -1:
        for i in range(len(headers) - 1, catalogIndex, -1):
            if re.search("[.]{4,} *\d+ *$", headers[i]):
                lastCatalogItemIndex = i
                break
    if lastCatalogItemIndex != -1:
        headers = headers[0:catalogIndex] + headers[lastCatalogItemIndex + 1:]

    return headers



def gen_outline(headers: list[str], trace: bool = False):
    global showTrace
    showTrace = trace

    outline = OutlineHeader("")
    prev: OutlineHeader = None

    for h in headers:
        hd_info = header_info(h)
        oh = OutlineHeader(h)
        included = False

        if prev is None:
            # append lv #1 header
            included = outline.add_child(oh)
        elif header_info(prev.label)['type'] == hd_info['type']:
            # add sibling
            included = prev.add_sibling(oh)
        else:
            # try to find parent in the same type
            proceeded = False
            for p in prev.parents:
                if header_info(p.label)['type'] == hd_info['type']:
                    # add sibling
                    included = p.add_sibling(oh)
                    proceeded = True
                    break

            # add as child if it doesn't match the type of any parents
            if proceeded == False:
                included = prev.add_child(oh)

        if included:
            prev = oh

    return outline


def header_info(label: str):
    typeIndex = -1
    order = ''

    # root header is empty
    if label != '':
        for i in range(len(exp_header_types)):
            m = re.match(exp_header_types[i], label)
            if m:
                typeIndex = i
                order = m.group(2)
                break

    return {
        "type": typeIndex if typeIndex != 0 else 1, # 0 and 1 are on the same level #1
        "order": int(cn2an.cn2an(order, 'smart')) if typeIndex > 0 and order != '' else 0 # the type index #0 isn't numeric
    }


catalogs = ""

def print_outline(outline: list[OutlineHeader], level: int = 0):
    global catalogs
    for h in outline:
        #print("    " * level + h.label)
        catalogs += "    " * level + h.label + "\n"
        print_outline(h.children, level + 1)

if __name__ == '__main__':
    path = "./data/赛伍技术.pdf"

    """
    from pyPDFStructure import PDFDocument

    fin = open(path, "rb")
    doc = PDFDocument(fin.read())
    fin.close()

    tree = doc.get_structure_tree()
    print(tree)
    """

    pages = load_pdf(path)
    headers = extract_headers(pages)
    outline = gen_outline(headers, True)

    print_outline(outline.children)

    if os.path.exists("catalogs.txt"):
        os.remove("catalogs.txt")

    with open('catalogs.txt', 'w') as f:
        f.write(catalogs)


    # pdf = pdfplumber.open(path)
    # page = pdf.pages[101]
    # tbls = page.extract_text_lines()
    # print(tbls)
