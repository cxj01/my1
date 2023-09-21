import os

from elasticsearch import Elasticsearch
from langchain import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from loader import UnstructuredPaddlePDFLoader
from textsplitter import AliTextSplitter
SPECIAL_FLAG = "#$%"

def load_file(filepath):
    with open(filepath, "r", encoding="utf-8")as f:
        data = f.readlines()
    docs = []
    for line in data:
        metadata = {}
        metadata["source"] = filepath
        metadata["filename"] = metadata["source"].split("/")[-1]
        metadata["file_directory"] = metadata["source"].replace("/" + metadata["filename"], "")
        metadata["filetype"] = "txt"
        metadata["page_number"] = 1
        metadata["category"] = "Title"
        temp_line = line.strip()
        element = metadata["filename"].replace(".txt", "") + "  " + SPECIAL_FLAG + temp_line
        docs.append(Document(page_content=str(element), metadata=metadata))
    return docs

index_name = "fg_test_en_pdf_bge"
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
# create_es_index(index=index_name)

# vc_name = "/root/.cache/torch/sentence_transformers/GanymedeNil_text2vec-large-chinese"
vc_name = "/root/.cache/torch/sentence_transformers/BAAI_bge-large-zh"
embeddings = HuggingFaceEmbeddings(model_name=vc_name,
                                                model_kwargs={'device': "cuda"})
vector_store = ElasticVectorSearch(
    elasticsearch_url=elasticsearch_url,
    index_name=index_name,
    embedding=embeddings,
    ssl_verify={"verify_certs": False})


dir_path = "/data/cxj_workplace/data/test_en_pdf_txt"
result = []
files = os.listdir(dir_path)
textsplitter = AliTextSplitter(pdf=True)
for file in files:
    filepath = os.path.join(dir_path, file)
    if os.path.isfile(filepath):
        docs = load_file(filepath)
        vector_store.add_documents(docs)
