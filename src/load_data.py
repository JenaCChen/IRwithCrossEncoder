import json
from domain_classification import generate_domain
import pymongo
from utils import timer

client = pymongo.MongoClient("localhost", 27017)
db = client["nfcorpus"]


def load_data(data_fname):
    """
    Loads corpus into a generator
    """
    with open(data_fname) as f:
        for key, line in enumerate(f):
            json_objs = json.loads(line)
            obj = {
                'id': key,
                '_id': json_objs['_id'],
                'title': json_objs['title'],
                'domain': generate_domain({json_objs["title"]}, {json_objs["text"]}),  # domain
                'content_str': json_objs['text'],
                'url': json_objs['metadata']['url']
            }
            yield obj

@timer
def insert_docs(docs):
    """
    create a collection 'nf_docs' on MongoDB database
    """
    db.create_collection('nf_docs')
    coll = db['nf_docs']
    coll.create_index([('id', pymongo.ASCENDING)], unique=True)
    coll.insert_many(docs)

@timer
def insert_queries(docs):
    if not "nf_queries" in db.list_collection_names():
        db.create_collection('nf_queries')
    coll = db['nf_queries']
    coll.create_index([('query_id', pymongo.ASCENDING)], unique=True)
    coll.insert_many(docs)


def load_queries(dataset: str = 'test'):
    queries = []
    scores = {}  # {'query_id': {'doc_id': score, ...}, ...}
    with open(f'data/nfcorpus/{dataset}.2-1-0.qrel') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] not in scores:
                scores[line[0]] = {}
            scores[line[0]][line[-2]] = int(line[-1])
    with open(f'data/nfcorpus/{dataset}.titles.queries') as f:
        for line in f:
            line = line.strip().split('\t')
            queries.append(
                {
                    'query_id': line[0],
                    'query_text': line[-1],
                    'query_type': dataset,
                    'score': scores.get(line[0], {}),
                }
            )
    return queries

# for data in ['train', 'dev', 'test']:
#     insert_queries(load_queries(data))
