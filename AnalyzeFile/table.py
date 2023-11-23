import tabula
from typing import Iterable, Optional, Union

def clean(x):
     if (type(x) == str):
        return x.replace('\r', ' ')
     else:
         return x

def read_tables(
    path: str,
    pages: Optional[Union[str, int, Iterable[int]]] = None,
):
    tables = tabula.read_pdf(path, pages=pages, lattice=True)

    for tbl in tables:
        tbl.columns = tbl.columns.map(clean)

    tables = [tbl.map(clean) for tbl in tables]

    return tables
