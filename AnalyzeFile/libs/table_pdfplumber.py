import pdfplumber
from pandas import DataFrame
from typing import Optional, List

def clean(tbl: List[List[str | None]]):
    for row in tbl:
        for index in range(0, len(row)):
            if row[index] != None:
                row[index] = row[index].replace('\r', ' ')
    
    return tbl

def read_tables(path: str, page: Optional[int] = None):
    pdf = pdfplumber.open(path)

    pdfPages = None
    if page == None:
        pdfPages = pdf.pages
    else:
        pdfPages = pdf.pages[page:page + 1]
    
    tables = {}
    for pg in pdfPages:
        tbls = [clean(t) for t in pg.extract_tables()]
        if len(tbls) > 0:
            tables[pg.page_number] = tbls

    return tables

def print_tables(pages: dict):
    for page, tables in pages.items():
        print(f"Page {page}:")

        for tbl in tables:
            print(DataFrame(tbl[1:], columns=tbl[0]))
            print("\n")
