from decimal import ROUND_HALF_UP, Context
import json
from pathlib import Path
from typing import List, Dict, Union, Generator, Tuple
from typing import Sequence, NamedTuple, List
import warnings
import math
import numpy as np  # type: ignore
import os
import pandas as pd
from search import no_domain_search


top_k = 10
top_k = 10
s_index = 'nf_docs'
data_dir = 'nfcorpus'

def _as_decimal(num: float) -> float:
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    return float(rounder.create_decimal_from_float(num * 100))


def load_query(nf_folder_path):
    """
    helper func that loads the test-set queries and yields the _id, _text as a tuple
    """
    nf_folder_path = Path(nf_folder_path)
    queries_path = nf_folder_path.joinpath('queries.jsonl')
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            tup = (json_obj['_id'], json_obj['text'])
            yield tup


def load_qrels(nf_split_file: Union[str, os.PathLike]) -> Dict[str, Dict]:
    # load the query relevance test file as a nested dictionary
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


def invert_gold_standard(gold_standard: Dict[str, Dict], k: int = top_k) -> Dict[str, List]:
    """
    helper func that returns a dictionary mapping query_id with to corresponding IDCG
    """
    invert_gold = {}  # {query_id: [1, 1, 1, 2, 2]}
    for doc_id in gold_standard:
        for query_id in gold_standard[doc_id]:
            if query_id not in invert_gold:
                invert_gold[query_id] = []
            invert_gold[query_id].append(gold_standard[doc_id][query_id])
    invert_gold = {query_id: sorted(invert_gold[query_id], reverse=True)[:k] for query_id in invert_gold}
    return invert_gold


def dcg(relevance: Sequence[int], k: int = 10) -> float:
    relevance_len = len(relevance)
    if relevance_len < k:
        warnings.warn(
            f"sequence length is smaller than k ({k})! Reset k to maximum sequence length ({relevance_len})",
            SyntaxWarning,
        )
    top_k = relevance[:k]
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(top_k, 1))


def ndcg(relevance: Sequence[int], idea_relevance: Sequence[int], k: int = 10) -> float:
    relevance_len = len(relevance)
    if relevance_len < k:
        warnings.warn(
            f"sequence length is smaller than k ({k})! Reset k to maximum sequence length ({relevance_len})",
            SyntaxWarning,
        )
    top_k = relevance[:k]
    ideal_relevance_len = len(idea_relevance)
    if ideal_relevance_len < k:
        idea_relevance += [0] * (k - ideal_relevance_len)
    idea_relevance = idea_relevance[:k]

    try:
        return dcg(top_k, k) / dcg(idea_relevance, k)
    except ZeroDivisionError:
        return 0.0


def get_score(query_id: str, matched_docs: List[str], gold_standard: Dict[str, Dict],
              gold_scores: List[int], k: int = top_k) -> float:
    """
    helper func that computes the NDCG score between the matched docs and the ideal/gold-standard docs
    """
    y_pred = [gold_standard.get(doc, {}).get(query_id, 0) for doc in matched_docs]
    y_pred.extend([0] * (k - len(y_pred)))  # pad the length to top_k just in case # of matched docs are less
    return _as_decimal(ndcg(y_pred, gold_scores, k))

def get_gold_score(query_id: str, inv_gold_standard: Dict[str, List], k: int = top_k) -> List[int]:
    """
    helper func that returns a list of the ideal scores given the param query_id, hence IDCG
    """
    y_true = inv_gold_standard[query_id]
    y_true.extend([0] * (k - len(y_true)))
    return y_true


def generate_cross_encoder_NDCG() -> list[float]:
    """
    Generate and save to file a list of the NDCG@10 scores of the cross-encoder model
    """
    queries = load_query(data_dir)
    gold_standard = load_qrels(Path.joinpath(data_dir, 'test.tsv'))
    inv_gold_standard = invert_gold_standard(gold_standard)
    scores = []
    print('Searching starts...')
    for (query_id, query_text) in queries:
        sc = get_score(
            query_id=query_id,
            matched_docs=no_domain_search(query_text, fine_tune=True),
            gold_standard=gold_standard,
            gold_scores=get_gold_score(query_id, inv_gold_standard)
        )
        scores.append(sc)
    with open('cross_encoder_scores_classification.csv', 'w') as f:
        f.write('ce_classification\n')
        for score in scores:
            f.write(f'{score}\n')


if __name__=='__main__':
    generate_cross_encoder_NDCG()