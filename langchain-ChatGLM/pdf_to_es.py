import json
import os

from elasticsearch import Elasticsearch
from langchain import ElasticVectorSearch
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

from loader import UnstructuredPaddlePDFLoader
from textsplitter import AliTextSplitter
from textsplitter.new_text_splitter import NewSpacyTextSplitter
SPECIAL_FLAG = "#$%"

def load_file(filepath, textsplitter):
    loader = PyPDFLoader(filepath)
    docs = loader.load_and_split(textsplitter)
    # for doc in docs:
    #     doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]
    #     doc.metadata["file_directory"] = doc.metadata["source"].replace("/" + doc.metadata["filename"], "")
    #     doc.metadata["filetype"] = "txt"
    #     doc.metadata["page_number"] = 1
    #     doc.metadata["category"] = "Title"
    for doc in docs:
        doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]
        doc.metadata["file_directory"] = doc.metadata["source"].replace("/" + doc.metadata["filename"], "")
        doc.metadata["filetype"] = "txt"
        doc.metadata["page_number"] = 1
        doc.metadata["category"] = "Title"
        # doc.page_content = ''.join(doc.metadata["filename"].split(".pdf")) + "  " + SPECIAL_FLAG + doc.page_content
    return docs
index_name = "fg_dmzc"
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


# dir_path = "/data/cxj_workplace/data/test_en_pdf/"
dir_path = "/data/cxj_workplace/data/长臂管辖/"
result = []
files = os.listdir(dir_path)
# textsplitter = AliTextSplitter(pdf=True)
# loader = PyPDFLoader(filepath)
textsplitter = NewSpacyTextSplitter(
                    pdf=True,
                    pipeline="zh_core_web_md",
                    separator=" ",
                    # pipeline="en_core_web_sm",
                    chunk_size=510,
                    chunk_overlap=120,
                )
error_data = []
for file in files:
    filepath = os.path.join(dir_path, file)
    try:
        if os.path.isfile(filepath):
            docs = load_file(filepath, textsplitter)
            if len(docs) < 10:
                error_data.append(filepath)
                print(filepath)
            else:
                vector_store.add_documents(docs)
    except Exception as e:
        print("***" * 10)
        print(filepath)
        error_data.append(filepath)
        print("***" * 10)

with open("error.txt", 'w', encoding="utf-8")as f:
    f.write(json.dumps({"error_data": error_data}, ensure_ascii=False, indent=4))
