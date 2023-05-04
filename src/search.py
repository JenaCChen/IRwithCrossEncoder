from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from load_data import insert_docs, load_data
import pymongo
import numpy as np
from utils import timer, get_device
import torch
from train import predict

device = get_device()
# model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device=device)

# fine-tuned model
model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = 'fine_tuned_crossEncoder.pth'
model = torch.load(model_path)

client = pymongo.MongoClient("localhost", 27017)
db = client["nfcorpus"]
coll = db['nf_docs']
corpus_path = 'data/nfcorpus-pa5/corpus.jsonl'


if not "nf_docs" in db.list_collection_names():
    insert_docs(load_data(corpus_path))


def no_domain_search(query: str, top_k: int = 10, fine_tune: bool = False):
    """
    Performs search without domain classes, used to compute NDCG scores
    """
    docs = coll.find({})
    ids = []
    prompts = []
    if docs is not None:
        for doc in docs:
            ids.append(doc['_id'])
            prompts.append([[query, f'{doc["title"]} {doc["content_str"]}']])
    scores = predict(device, prompts) if fine_tune else model.predict(prompts)
    # best_idx = np.argsort(scores)[-top_k:]  # top-k matches, obsolete
    best_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    best_idx = [i for i, _ in best_idx]
    # print([scores[idx] for idx in best_idx])
    return [ids[idx] for idx in best_idx]

@timer
def search(query: str, domain: str, top_k: int = 10):
    """
    Returns a list of top k matched doc ids
    """
    domain_docs = coll.find({'domain': domain})  # generator
    ids = []
    prompts = []
    if domain_docs is not None:
        for doc in domain_docs:
            ids.append(doc['_id'])
            prompts.append((query, f'{doc["title"]} {doc["content_str"]}'))
    scores = model.predict(prompts)
    best_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k] # top-k matches
    best_idx = [i for i, _ in best_idx]
    return [ids[idx] for idx in best_idx]