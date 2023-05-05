import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pymongo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils import get_device
import random
from collections import Counter
from load_data import doc_lookup, query_lookup

random.seed(10)
client = pymongo.MongoClient("localhost", 27017)
db = client["nfcorpus"]

# Used for re-scaling the relevancy score
scaled_label = {2: 10, 1: 1, 0: 0}
# scaled_label = {2: 1, 1: 0.1, 0: 0}

lr = 5e-6
epoch = 3
batch = 8.
model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = 'cross_encoder_balanced_recaled.pt'
device = get_device()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


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

def compile_train_set(limit: int = 323):
    """
    Compiles training set with data augmentation and label balancing, the param defines the number queries sampled
    """
    docs_coll = db['nf_docs']
    query_coll = db['nf_queries']

    # randomly sampled the param number of queries
    queries = query_coll.aggregate([
        {"$match": {'query_type': 'train'}},
        {"$sample": {"size": limit}}
    ])

    sent_contents, sent_labels, label_one_contents, label_one_labels = [], [], [], []
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
                        content = [[query["query_text"], f'{relevant_doc["title"]} {relevant_doc["content_str"]}']]
                        # differentiating relevancy score 1
                        if score == 1:
                            label_one_contents.append(content)
                            label_one_labels.append(scaled_label[score])
                        else:
                            sent_contents.append(content)
                            sent_labels.append(scaled_label[score])
    assert len(sent_contents) == len(sent_labels)
    assert len(label_one_contents) == len(label_one_labels)
    # data balancing
    if len(label_one_contents) > len(sent_contents):
        sent_tup = list(zip(label_one_labels, label_one_contents))
        sampled_sent_tup = random.sample(sent_tup, int(len(sent_contents) * 0.6))
        sent_contents.extend([content for (_, content) in sampled_sent_tup])
        sent_labels.extend([label for (label, _) in sampled_sent_tup])
    assert len(sent_contents) == len(sent_labels)

    print(f'{limit} numbers of queries sampled, total of {len(sent_contents)} documents included.')
    label_counts = Counter(sent_labels)
    for label in label_counts:
        print(f'There are {label_counts[label]} data entries of relevancy score {label}.')
    return sent_contents, sent_labels


def train(model, optimizer, loss_fn, device, train_dataset, epoch, model_path):
    """
    Train and save the model
    """
    model = model.to(device)
    print(device)
    model.train()
    print_loss_freq = 8
    for i in range(epoch):
        epoch_loss = 0
        for j, (input_ids, attention_masks, labels) in enumerate(train_dataset):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_masks)
            loss = loss_fn(output.logits.view(-1), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if j % epoch == 0:
                print(round(epoch_loss / print_loss_freq, 2))
                epoch_loss = 0
    torch.save(model.state_dict(), model_path)


def inference(device, test_dataset):
    """
    Make inferences with the model trained
    """
    model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    scores = []
    for input_ids, attention_masks in test_dataset:
        output = model(input_ids=input_ids, attention_mask=attention_masks).logits.view(-1)
        scores.extend(output.tolist())
    return scores


def predict(device, prompts: list):
    """
    Make predictions given the param list of (query, document) input,
    for server use
    """
    test_dataset = TextDataset(prompts)
    test_dataset = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    scores = inference(device, test_dataset)
    return scores


if __name__ == '__main__':
    contents, labels = compile_train_set(limit=1000)
    train_dataset = TextDataset(contents, labels)
    train_dataset = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    train(model, optimizer, loss_fn, device, train_dataset, epoch, model_path)
