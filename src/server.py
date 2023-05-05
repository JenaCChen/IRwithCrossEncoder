from flask import Flask, request
from flask_cors import CORS
from elasticsearch_dsl.connections import connections
from search import boolean_search, tf_idf_search, bi_encoder_search, rerank_cross_encoder, ec_search, query_classification
from utils import timer


app = Flask(__name__)
CORS(app)

@timer
def boolean_retrieval(index_name, query, domain):
    domain = map_domain(domain)
    document_list = []
    response = boolean_search(index_name, query, domain)
    for hit in response:
        hit_dict = {
            "id": hit.doc_id,
            "title": hit.title,
            "content": hit.content
        }
        document_list.append(hit_dict)
    boolean_list = generate_list("boolean", document_list)
    return boolean_list

@timer
def tf_idf_retrieval(index_name, query, domain):
    domain = map_domain(domain)
    query = strip_query(query)
    response = tf_idf_search(index_name, query, domain)
    document_list = []
    for hit in response:
        hit_dict = {
            "id": hit.doc_id,
            "title": hit.title,
            "content": hit.content
        }
        document_list.append(hit_dict)
    tf_idf_list = generate_list("tf_idf", document_list)

    return tf_idf_list

@timer
def bi_encoder_retrieval(index_name, query, domain):
    domain = map_domain(domain)
    query = strip_query(query)
    response = bi_encoder_search(index_name, query, domain)
    document_list = []
    for hit in response:
        hit_dict = {
            "id": hit.doc_id,
            "title": hit.title,
            "content": hit.content
        }
        document_list.append(hit_dict)
    bi_encoder_list = generate_list("bi_encoder", document_list)
    return bi_encoder_list


def map_domain(domain):
    mapping = {
        "news and social concern": "news_&_social_concern",
        "fitness and health": "fitness_&_health",
        "food and dining": "food_&_dining",
        "science and technology": "science_&_technology",
        "other": "other_hobbies"
    }
    return mapping[domain]


def strip_query(query):
    query_split = query.split()
    new_query_list = []
    for q in query_split:
        if q[0] in ["+", "-", "?"]:
            new_query_list.append(q[1:])
        else:
            new_query_list.append(q)
    return " ".join(new_query_list)


def generate_list(approach_name, document_list):
    list_for_retrieval_method = [
        {
            "approach": approach_name,
            "documents": document_list
        }
    ]
    return list_for_retrieval_method

@app.route('/result', methods = ['POST', 'GET'])
def get_result():
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    index_name = "es_corpus"
    query = request.args.get('query')
    domain = request.args.get('domain')
    approach = request.args.get('approach')

    # query_class = query_classification(query)  # obsolete due to retrieval time constraint
    boolean_result = boolean_retrieval(index_name, query, domain)
    tf_idf_result = tf_idf_retrieval(index_name, query, domain)
    bi_encoder_result = bi_encoder_retrieval(index_name, query, domain)
    # ec_result = ec_search(query=query, domain=map_domain(domain))  # obsolete due to retrieval time constraint
    ec_rerank_result = rerank_cross_encoder(query=query, domain=map_domain(domain), index=index_name)
    ec_result = ec_rerank_result



    if approach == "Boolean":
        return [boolean_result[0], tf_idf_result[0], bi_encoder_result[0], ec_result[0]]
    elif approach == "TF-IDF":
        return [tf_idf_result[0], boolean_result[0], bi_encoder_result[0], ec_result[0]]
    elif approach == "Bi-Encoder":
        return [bi_encoder_result[0], tf_idf_result[0], boolean_result[0], ec_result[0]]
    elif approach == 'Cross-Encoder':
        return [ec_result[0], tf_idf_result[0], boolean_result[0], bi_encoder_result[0]]


if __name__ == "__main__":
    app.run(debug=True)
