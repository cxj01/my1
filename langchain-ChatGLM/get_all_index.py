from elasticsearch import Elasticsearch
ELASTICSEARCH_URL = f"http://10.1.226.1:9220"
es_client = Elasticsearch(hosts=[ELASTICSEARCH_URL], verify_certs=False)
r = es_client.cat.indices(index="fg_*", format="json")
print(12)