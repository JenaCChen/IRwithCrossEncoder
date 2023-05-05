from elasticsearch_dsl import Q, Search
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
from embedding_service.client import EmbeddingClient
from sentence_transformers import CrossEncoder
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match
import pymongo
from utils import timer, get_device
from load_data import insert_queries, load_queries, insert_docs, load_data
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

query_label_mapping = {
    0: 'BM25',
    1: 'BM25+bi-encoder',
    2: 'ce_balance_rescaled',
}
device = get_device()
model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device=device)  # corss-encoder model

# query classification model
query_model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=len(query_label_mapping))
query_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
query_model.load_state_dict(torch.load('query_classifier.pt'))

# mongoDB
client = pymongo.MongoClient("localhost", 27017)
db = client["nfcorpus"]
coll = db['nf_docs']


# create MongoDB collections if not exist already
if not "nf_docs" in db.list_collection_names():
    insert_docs(load_data(Path.joinpath(Path(__file__), 'nfcorpus', 'corpus.jsonl')))

if not 'nf_queries' in db.list_collection_names():
    for data in ['train', 'dev', 'test']:
        insert_queries(load_queries(data))


# the input looks like: +i +like -eating ?cake
def parse_query(query):
    must_list = []
    should_list = []
    not_list = []
    for token in query.split():
        signal = token[0]
        word = token[1:]
        if signal == "+":
            must_list.append(word)
        elif signal == "-":
            not_list.append(word)
        elif signal == "?":
            should_list.append(word)
        else:
            must_list.append(token)
    return must_list, should_list, not_list


def generate_Q(must_list, should_list, not_list):
    must_Q = []
    should_Q = []
    not_Q = []
    for must in must_list:
        must_Q.append(Q("match", full_content=must))
    for should in should_list:
        should_Q.append(Q("match", full_content=should))
    for n in not_list:
        not_Q.append(Q("match", full_content=n))
    return must_Q, should_Q, not_Q


def boolean_search(index_name, search_query, domain):
    must_list, should_list, not_list = parse_query(search_query)
    must_Q, should_Q, not_Q = generate_Q(must_list, should_list, not_list)
    if domain:
        must_Q.append(Q("match", domain=domain))
    search_query = Q('bool', must=must_Q, should=should_Q, must_not=not_Q)
    s = Search(using="default", index=index_name).query(search_query)
    response = s.execute()
    return response[:10]

    # query_domain = Match(domain=domain)
    # s = Search(using="default", index=index_name).query(query_text & query_domain)


def tf_idf_search(index_name, search_query, domain):
    query_text = Match(full_content={"query": search_query})
    query_domain = Match(domain=domain)
    s = Search(using="default", index=index_name).query(query_text & query_domain)
    response = s.execute()
    return response[:10]


def bi_encoder_search(index_name, search_query, domain):
    encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    query_vector = encoder.encode([search_query]).tolist()[0]
    q_vector = generate_script_score_query(
        query_vector, "sbert_embedding"
    )
    query_domain = Match(domain=domain)
    response = Search(using="default", index=index_name).query(q_vector & query_domain)
    return response[:10]


def generate_script_score_query(query_vector, vector_name):
    q_script = ScriptScore(
        query={"match_all": {}},
        script={
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
            # add 1.0 to avoid negative score
            "params": {"query_vector": query_vector},
        },
    )
    return q_script


def no_domain_search(query: str, top_k: int = 10) -> list[str]:
    """
    Search with off-the-shelf Cross-encoder ranking and returns a list of top k matched doc ids
    Used for scoring only
    """
    all_docs = coll.find({})  # generator
    doc_ids, prompts = [], []
    if all_docs is not None:
        for doc in all_docs:
            doc_ids.append(doc['_id'])
            prompts.append((query, f'{doc["title"]} {doc["content_str"]}'))
    scores = model.predict(prompts)
    best_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]  # top-k matches
    best_idx = [i for i, _ in best_idx]
    return [doc_ids[idx] for idx in best_idx]

@timer
def ec_search(query: str, domain: str, top_k: int = 10) -> list[dict]:
    """
    Search with off-the-shelf Cross-encoder ranking,
    returns a list of a dict containing the search approach and a list of top k matched doc ids
    """
    domain_docs = coll.find({'domain': domain})  # generator
    ids, prompts, titles, contents = [], [], [], []
    if domain_docs is not None:
        for doc in domain_docs:
            ids.append(doc['_id'])
            prompts.append((query, f'{doc["title"]} {doc["content_str"]}'))
            titles.append(doc['title'])
            contents.append(doc['content_str'])
    scores = model.predict(prompts)
    best_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]  # top-k matches
    best_idx = [i for i, _ in best_idx]
    best_docs = list(
        zip([ids[idx] for idx in best_idx], [titles[idx] for idx in best_idx], [contents[idx] for idx in best_idx]))
    output = [
        {
            'approach': 'cross-encoder',
            'documents': [
                {'id': best_docs[i][0], 'title': best_docs[i][1], 'content': best_docs[i][2]} for i in
                range(len(best_docs))
            ],
        },
    ]
    return output

@timer
def rerank_cross_encoder(query: str, domain: str, index: str, top_k: int = 10) -> list[dict]:
    """
    Search with BM25 and rerank with off-the-shelf Cross-encoder,
    returns a list of a dict containing the search approach and a list of top k matched doc ids
    """
    q_bm25 = Match(full_content={'query': query})
    query_domain = Match(domain=domain)
    s = Search(using='default', index=index).query(q_bm25 & query_domain)[:25]
    response = s.execute()
    ids, prompts, titles, contents = [], [], [], []
    for hit in response:
        prompts.append((query, f'{hit.title} {hit.content}'))
        ids.append(hit.doc_id)
        titles.append(hit.title)
        contents.append(hit.content)
    if len(ids) > 0:
        scores = model.predict(prompts)
        best_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]  # top-k matches
        best_idx = [i for i, _ in best_idx]
        best_docs = list(
            zip([ids[idx] for idx in best_idx], [titles[idx] for idx in best_idx], [contents[idx] for idx in best_idx]))
        output = [
            {
                'approach': 'cross-encoder',
                'documents': [
                    {'id': best_docs[i][0], 'title': best_docs[i][1], 'content': best_docs[i][2]} for i in
                    range(len(best_docs))
                ],
            },
        ]
        return output
    else:
        return [
            {
                'approach': 'cross-encoder',
                'documents': [],
            },
        ]


def query_classification(query: str) -> list[dict[str, str]]:
    """
    Given the param query, returns the best retrieval approach suggested by the classifier model
    """
    tokens = query_tokenizer.encode_plus(
        query,
        max_length=16,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    logits = query_model(**tokens).logits
    pred = torch.argmax(logits, dim=-1)
    pred = query_label_mapping[pred.item()]
    return [
            {
                'best_approach': pred,
            },
        ]