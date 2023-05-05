from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import pandas as pd
import json
import random
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from pathlib import Path
from utils import get_device

random.seed(42)
label_mapping = {
    'BM25': 0,
    'BM25+bi-encoder': 1,
    'ce_balance_rescaled': 2,
}


def compile_dataset():
    """
    Reads, split, and returns the training and dev sets
    """
    query_data = {}
    queries_map = {}
    with open(Path.joinpath(Path(__file__), 'nfcorpus', 'queries.jsonl')) as query_file:
        for val, line in enumerate(query_file):
            temp = json.loads(line)
            query_data[temp['_id']] = temp['text']
            query_data[line[0]] = line[-1]
            queries_map[val] = temp['_id']

    queries = pd.read_csv('merged_score.csv', sep='\t')
    queries = queries.rename(queries_map)

    data_by_label = defaultdict(list)
    for i, row in queries.iterrows():
        data_by_label[row[-1]].append(query_data[i])
    data_by_label['BM25'] = random.sample(data_by_label['BM25'], int(len(data_by_label['BM25+bi-encoder']) * 1.1))
    data = [(query, label_mapping[label]) for label in data_by_label for query in data_by_label[label]]
    random.Random(42).shuffle(data)
    split_point = int(len(data) * 0.9)
    return data[:split_point], data[split_point:]


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens = tokenizer(
            self.data[index][0],
            add_special_tokens=True,
            max_length=16,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        label = torch.tensor(self.data[index][1], dtype=torch.int32)
        return input_ids.to(device), attention_mask.to(device), label.to(device)


def train(model, optimizer, loss_fn, device, train_dataset, dev_dataset, epoch, model_path):
    """
    Train the query classification model with accuracy on the dev set returned
    """
    model = model.to(device)
    model.train()
    print_loss_freq = 32
    for i in range(epoch):
        epoch_loss = 0
        for j, (input_ids, attention_masks, labels) in enumerate(train_dataset):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_masks)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if j % epoch == 0:
                print(round(epoch_loss / print_loss_freq, 4))
                epoch_loss = 0
    torch.save(model.state_dict(), model_path)
    total_correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_masks, labels in dev_dataset:
            output = model(input_ids=input_ids, attention_mask=attention_masks).logits
            preds = torch.argmax(output, dim=-1)
            total_correct += (preds == labels).sum()
            total += len(labels)
    acc = total_correct / total
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    device = get_device
    model_name = 'roberta-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lr = 5e-6
    epoch = 5
    batch = 12
    model_path = 'query_classifier.pt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_data, dev_data = compile_dataset()
    train_set = TextDataset(train_data)
    dev_set = TextDataset(dev_data)
    train_set = DataLoader(train_set, batch_size=batch, shuffle=True)
    dev_set = DataLoader(dev_set, batch_size=batch, shuffle=False)
    train(model, optimizer, loss_fn, device, train_set, dev_set, epoch, model_path)
