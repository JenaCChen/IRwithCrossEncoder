import json
from pathlib import Path
import numpy as np
import pandas as pd
import json
from domain_classification import generate_domain
import pymongo
from utils import timer

client = pymongo.MongoClient("localhost", 27017)
db = client["nfcorpus"]
current_dir = Path(__file__)
data_dir = 'nfcorpus'


def load_qrels(nf_split_file):
	qrels_df = pd.read_csv(nf_split_file, sep="\t", header=0)
	docs_rels_dict = dict()
	for i, group in enumerate(qrels_df.groupby("corpus-id")):
		doc_id = group[0]
		doc_rels_dict = dict()
		group_df = group[1][["query-id", "score"]]
		for _, row in group_df.iterrows():
			doc_rels_dict[row["query-id"]] = row["score"]
		docs_rels_dict[doc_id] = doc_rels_dict
	return docs_rels_dict

def load_nf_corpus(nf_path):
	nf_path = Path(nf_path)
	nf_docs_path = nf_path.joinpath("corpus_with_domain.jsonl")
	nf_split_path = nf_path.joinpath("test.tsv")
	nf_docs_embeds_path = nf_path.joinpath("nf_docs_sb_mp_net_embedddings.npy")

	nf_docs_embeddings = np.load(str(nf_docs_embeds_path))
	docs_rels_dict = load_qrels(nf_split_path)
	with open(nf_docs_path, "r", encoding="utf-8") as f:
		for index, line in enumerate(f):
			doc_dict = json.loads(line)
			doc_dict["annotation"] = docs_rels_dict.get(doc_dict["id"], dict())
			doc_dict["sbert_embedding"] = nf_docs_embeddings[index].tolist()
			yield doc_dict

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
    """
    create a collection 'nf_queries' on MongoDB database
    """
    if not "nf_queries" in db.list_collection_names():
        db.create_collection('nf_queries')
    coll = db['nf_queries']
    coll.create_index([('query_id', pymongo.ASCENDING)], unique=True)
    coll.insert_many(docs)


def load_queries(dataset: str = 'test') -> dict[dict[str, int]]:
    """
    Reads the param query set in nf-corpus returns a dict of dict.
    {'query_id': {'doc_id': score, ...}, ...}
    """
    queries = []
    scores = {}  # {'query_id': {'doc_id': score, ...}, ...}
    with open(Path.joinpath(current_dir, data_dir, f'{dataset}.2-1-0.qrel')) as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] not in scores:
                scores[line[0]] = {}
            scores[line[0]][line[-2]] = int(line[-1])
    with open(Path.joinpath(current_dir, data_dir, f'{dataset}.titles.queries')) as f:
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


def doc_lookup(doc_id):
    """
    helper function
    """
    coll = db['nf_docs']
    doc = coll.find({'_id': doc_id})
    return doc


def query_lookup(query_id):
    coll = db['nf_queries']
    query = coll.find({'query_id': query_id})
    return query