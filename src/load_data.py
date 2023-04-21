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
