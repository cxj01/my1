from abc import ABC
from typing import Optional, Dict, Any, List, Tuple, Type

from langchain import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.vectorstores.base import VST


def _my_script_query(query_vector, query, query_vector2, query2):
    return {
        "script_score": {
            "query": {
                "more_like_this": {
                  "fields": [
                    "text"
                  ],
                  "like": query,
                  "min_term_freq": 1,
                  "max_query_terms": 12,
                  "min_doc_freq": 1
                }
                # "match_all": {}
            },
            "script": {
                "source": "_score + cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector},
            },
        }
    }
    # return {
    #     "script_score": {
    #         "query": {
    #             "bool": {
    #                 "minimum_should_match": 1,
    #                 "should": [
    #                     {
    #                         "more_like_this": {
    #                             "fields": [
    #                                   "text"
    #                                 ],
    #                         "like": query,
    #                         "min_term_freq": 1,
    #                         "max_query_terms": 12,
    #                         "min_doc_freq": 1
    #                         }
    #                     },
    #                     {
    #                           "more_like_this": {
    #                                 "fields": [
    #                                     "text"
    #                                 ],
    #                             "like": query2,
    #                             "min_term_freq": 1,
    #                             "max_query_terms": 12,
    #                             "min_doc_freq": 1
    #                       }
    #                     }
    #                   ]
    #             }
    #         },
    #         "script": {
    #             "source": "_score + cosineSimilarity(params.query_vector, 'vector') + cosineSimilarity(params.query_vector2, 'vector') + 1.0",
    #             "params": {
    #                 "query_vector": query_vector,
    #                 "query_vector2": query_vector2
    #             },
    #         },
    #     }
    # }

def _all_script_query(query_vector, query, file_list):
    return {
        "script_score": {
            "query": {
                "bool": {
                    "must": [
                        # {
                        #     "more_like_this": {
                        #         "fields": [
                        #             "text"
                        #         ],
                        #         "like": query,
                        #         "min_term_freq": 1,
                        #         "max_query_terms": 12,
                        #         "min_doc_freq": 1
                        #     }
                        # },
                        {
                            "terms": {
                                "metadata.filename": file_list
                            }
                        }
                    ]
                }
            },
            "script": {
                "source": "_score + cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {
                    "query_vector": query_vector
                }
            }
        }
    }

class MyElasticVectorSearch(ElasticVectorSearch, ABC):
    def __init__(self,
        elasticsearch_url: str,
        index_name: str,
        embedding: Embeddings,
        ssl_verify: Optional[Dict[str, Any]] = None,):
        super().__init__(elasticsearch_url=elasticsearch_url,
                       index_name=index_name,
                       embedding=embedding,
                       ssl_verify=ssl_verify)

    def similarity_search_with_score(
        self, query: str, query2: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        embedding2 = self.embedding.embed_query(query2)
        # embedding = self.embedding.encode(query)
        script_query = _my_script_query(embedding, query, embedding2, query2)
        response = self.client_search(
            self.client, self.index_name, script_query, size=k
        )
        hits = [hit for hit in response["hits"]["hits"]]
        docs_and_scores = [
            (
                Document(
                    page_content=hit["_source"]["text"],
                    metadata=hit["_source"]["metadata"],
                ),
                hit["_score"],
            )
            for hit in hits
        ]
        return docs_and_scores

    def all_search_with_score(
        self, query: str, k: int = 10000, file_list: List = []
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        # embedding = self.embedding.encode(query)
        script_query = _all_script_query(embedding, query, file_list)
        response = self.client_search(
            self.client, self.index_name, script_query, size=k
        )
        hits = [hit for hit in response["hits"]["hits"]]
        docs_and_scores = [
            (
                Document(
                    page_content=hit["_source"]["text"],
                    metadata=hit["_source"]["metadata"],
                ),
                hit["_score"],
            )
            for hit in hits
        ]
        return docs_and_scores





# vc_name = "/root/.cache/torch/sentence_transformers/GanymedeNil_text2vec-large-chinese"
# embeddings = HuggingFaceEmbeddings(model_name=vc_name,
#                                                 model_kwargs={'device': "cuda"})
# elastic_host = "10.1.226.1"
# elasticsearch_url = f"http://{elastic_host}:9220"
# vector_store = MyElasticVectorSearch(
#     elasticsearch_url=elasticsearch_url,
#     index_name="fg_ik",
#     embedding=embeddings,
#     ssl_verify={"verify_certs": False},
# )
# query = "刑法第十五条的内容"
# related_docs_with_score = vector_store.similarity_search_with_score(query, k=5)
# print(12)

