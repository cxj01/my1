import os

from langchain.document_loaders import TextLoader

from textsplitter import AliTextSplitter


def load_file(filepath):
    loader = TextLoader(filepath, autodetect_encoding=True)
    textsplitter = AliTextSplitter(pdf=True)
    docs = loader.load_and_split(textsplitter)
    return docs

dir_path = "/data/cxj_workplace/data/xw_ew"
files = os.listdir(dir_path)
for file in files:
    if os.path.isfile(os.path.join(dir_path, file)):
        docs = load_file(os.path.join(dir_path, file))
        print(123)
