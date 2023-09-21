import glob
import os

import markdown
from elasticsearch import Elasticsearch
from langchain import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from loader import UnstructuredPaddlePDFLoader
from textsplitter import AliTextSplitter
SPECIAL_FLAG = "#$%"

def judge_single(text):
    text = text.replace(" ", "")
    if text[0] == "第" and "条" in text[0:7]:
        return True
    else:
        return False


def md_add_info(md_file):
    file_name = md_file.split("/")[-1]
    result = []
    with open(md_file, "r", encoding="utf-8") as file:
        content = file.read()
    html = markdown.markdown(content)
    heading_levels = []
    pre_level = {}
    latest_level = 1
    pre_element = ""
    for line in html.split("\n"):
        try:
            line_text = line.split(">")[1].split("</")[0]
        except:
            continue
        if line_text != "":
            if line.startswith("<h"):
                heading_level = int(line[2])
                pre_level[heading_level] = line_text
                heading_levels.append(heading_level)
                latest_level = heading_level
            if line.startswith("<p"):
                if judge_single(line_text) and pre_element != "":
                    element = file_name.split(".md")[0] + "  " + "  ".join(list(dict(sorted(pre_level.items())).values())[0:latest_level]) + "   " + SPECIAL_FLAG + pre_element
                    metadata = {
                        "source": md_file,
                        "filename": file_name,
                        "file_directory": md_file.split("/" + file_name)[0],
                        "filetype": "text/markdown",
                        "page_number": 1,
                        "category": "Title"
                    }
                    result.append(Document(page_content=str(element), metadata=metadata))
                    pre_element = line_text
                else:
                    pre_element += line_text
    if pre_element != "":
        element = file_name.split(".md")[0] + "  " + "  ".join(
            list(dict(sorted(pre_level.items())).values())[0:latest_level]) + "   " + SPECIAL_FLAG + pre_element
        metadata = {
            "source": md_file,
            "filename": file_name,
            "file_directory": md_file.split("/" + file_name)[0],
            "filetype": "text/markdown",
            "page_number": 1,
            "category": "Title"
        }
        result.append(Document(page_content=str(element), metadata=metadata))
    return result

# index_name = "fg_zlh_jl"
index_name = "fg_zlh_jl_bge"
elasticsearch_url = f"http://10.1.226.1:9220"
def create_es_index(index):
    mappings = {
        "properties": {
          "metadata": {
            "properties": {
              "category": {
                "type": "keyword"
              },
              "file_directory": {
                "type": "keyword"
              },
              "filename": {
                "type": "keyword"
              },
              "filetype": {
                "type": "keyword"
              },
              "page_number": {
                "type": "long"
              },
              "source": {
                "type": "keyword"
              }
            }
          },
          "text": {
            "type": "text",
            "analyzer": "ik_smart"
          },
          "vector": {
            "type": "dense_vector",
            "dims": 1024,
            "index": False
          }
    }
    }
    settings = {
    "analysis": {
      "analyzer": {
        "ik_smart": {
          "type": "custom",
          "tokenizer": "ik_smart"
        },
        "ik_max_word": {
          "type": "custom",
          "tokenizer": "ik_max_word"
        }
      }
    }
  }
    es_client = Elasticsearch(hosts=[elasticsearch_url], verify_certs=False)
    es_client.indices.create(index=index, mappings=mappings, settings=settings)

create_es_index(index=index_name)

# vc_name = "/root/.cache/torch/sentence_transformers/GanymedeNil_text2vec-large-chinese"
vc_name = "/root/.cache/torch/sentence_transformers/BAAI_bge-large-zh"
embeddings = HuggingFaceEmbeddings(model_name=vc_name,
                                                model_kwargs={'device': "cuda"})
vector_store = ElasticVectorSearch(
    elasticsearch_url=elasticsearch_url,
    index_name=index_name,
    embedding=embeddings,
    ssl_verify={"verify_certs": False})

dir_path = "/data/cxj_workplace/learning/Laws"
files = glob.glob("/data/cxj_workplace/learning/Laws/**/*.md")
# dir_path = "/data/cxj_workplace/data/md_file/"
# result = []
# files = os.listdir(dir_path)
textsplitter = AliTextSplitter(pdf=True)
for file in files:
    if "index.md" not in file and "修正案" not in file:
        if os.path.isfile(file):
            docs = md_add_info(file)
            # print(12)
            vector_store.add_documents(docs)
