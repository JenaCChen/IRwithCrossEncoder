import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import CrossEncoder
import pymongo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils import get_device
import random

random.seed(10)

client = pymongo.MongoClient("localhost", 27017)
db = client["nfcorpus"]

# scaled_label = {2: 10, 1: 1, 0: 0}
scaled_label = {2: 1, 1: 0.1, 0: 0}


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = tokenizer(
            self.texts[index],
            add_special_tokens=True,
            max_length=514,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.float32)
            return input_ids.to(device), attention_mask.to(device), label.to(device)
        else:
            return input_ids.to(device), attention_mask.to(device)


def doc_lookup(doc_id):
    coll = db['nf_docs']
    doc = coll.find({'_id': doc_id})
    return doc


def query_lookup(query_id):
    coll = db['nf_queries']
    query = coll.find({'query_id': query_id})
    return query


def compile_dataset(limit: int = 0, dataset_type: str = 'test'):
    sent_contents = []
    sent_labels = []

    docs_coll = db['nf_docs']
    # docs = docs_coll.find()
    # all_docs_ids = [doc['_id'] for doc in docs]

    query_coll = db['nf_queries']
    if limit > 0:
        # # randomly sampled the param number of queries
        queries = query_coll.aggregate([
            {"$match": {'query_type': dataset_type}},
            {"$sample": {"size": limit}}
        ])  # random
    else:
        queries = query_coll.find({'query_type': dataset_type})
    # if queries is not None:
    #     for query in queries:
    #         scores = query['score']
    #         for doc_id in all_docs_ids:
    #             docs = doc_lookup(doc_id)
    #             if docs is not None:
    #                 for doc in docs:
    #                     sent_contents.append([[query["query_text"], f'{doc["title"]} {doc["content_str"]}']])
    #                     score = scores.get(doc_id, 0)
    #                     sent_labels.append(scaled_label[score])
    if queries is not None:
        for query in queries:
            rel_docs = query['score']  # # {'doc_id': score, ...}
            # # randomly sampled a number of (0, 5) of irrelevant documents
            irre_docs = docs_coll.aggregate([{"$sample": {"size": random.randint(0, 5)}}])
            for irre_doc in irre_docs:
                rel_docs[irre_doc['_id']] = rel_docs.get(irre_doc['_id'], 0)
            for (relevant_doc_id, score) in rel_docs.items():
                rel_docs = doc_lookup(relevant_doc_id)
                if rel_docs is not None:
                    for relevant_doc in rel_docs:
                        sent_contents.append(
                            [[query["query_text"], f'{relevant_doc["title"]} {relevant_doc["content_str"]}']]
                        )
                        sent_labels.append(scaled_label[score])
    assert len(sent_contents) == len(sent_labels)
    print(f'{limit} numbers of queries sampled, total of {len(sent_contents)} documents included.')
    return sent_contents, sent_labels


contents, labels = compile_dataset(dataset_type='train', limit=323)

from collections import Counter
print(Counter(labels))
raise ValueError()
train_dataset = TextDataset(contents, labels)
# # dev_dataset = TextDataset(compile_trainset(dataset_type='dev'))
# # test_dataset = TextDataset(compile_trainset(dataset_type='test'))


lr = 1e-5
epoch = 5
batch = 2
model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = 'fine_tuned_crossEncoder_rescaled.pth'
device = get_device()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()  # MSE

train_dataset = DataLoader(train_dataset, batch_size=batch, shuffle=True)
# # dev_dataset = DataLoader(dev_dataset, batch_size=batch, shuffle=True)
# # test_dataset = DataLoader(test_dataset, batch_size=batch, shuffle=True)



def train(model, optimizer, loss_fn, device, train_dataset, epoch, model_path):
    model = model.to(device)
    print(device)
    model.train()
    for i in range(epoch):
        epoch_loss = 0
        for j, (input_ids, attention_masks, labels) in enumerate(train_dataset):
            output = model(input_ids=input_ids, attention_mask=attention_masks)
            loss = loss_fn(output.logits.view(-1), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {i + 1} out of {epoch}: Loss {epoch_loss}')
    torch.save(model, model_path)

# train(model, optimizer, loss_fn, device, train_dataset, epoch, model_path)


def inference(model, device, test_dataset):
    model.to(device)
    model.eval()
    scores = []
    for input_ids, attention_masks in test_dataset:
        output = model(input_ids=input_ids, attention_mask=attention_masks).logits.view(-1)
        scores.extend(output.tolist())
    return scores


def predict(model, device, batch, prompts: list):
    test_dataset = TextDataset(prompts)
    test_dataset = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    scores = inference(model, device, test_dataset)
    return scores
