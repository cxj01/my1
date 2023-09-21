import os

from loader import UnstructuredPaddlePDFLoader
from textsplitter import ChineseTextSplitter, AliTextSplitter
import json

dir_path = "/data/cxj_workplace/data/俄乌/"
result = []
files = os.listdir(dir_path)
textsplitter = AliTextSplitter(pdf=True)
for file in files:
    filepath = os.path.join(dir_path, file)
    if os.path.isfile(filepath):
        loader = UnstructuredPaddlePDFLoader(filepath)
        # textsplitter = ChineseTextSplitter(pdf=True, sentence_size=100)
        docs = loader.load_and_split(textsplitter)

        for doc in docs:
            # metadata = doc.metadata
            doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]
            doc.metadata["file_directory"] = doc.metadata["source"].replace("/" + doc.metadata["filename"], "")
            doc.metadata["filetype"] = "txt"
            doc.metadata["page_number"] = 1
            doc.metadata["category"] = "Title"

            result.append({
                "page_content": doc.page_content,
                "meta_data": doc.metadata
            })

with open("ew_2.json", "w", encoding="utf-8")as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=4))