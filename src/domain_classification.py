import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/tweet-topic-21-multi")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
label_map = {
    7: 'fitness_&_health',
    8: 'food_&_dining',
    12: 'news_&_social_concern',
    15: 'science_&_technology'
}


def generate_domain(doc_title, doc_content):
    """
    Returns the domain class given
    """
    inputs = tokenizer(f'{doc_title} {doc_content}', return_tensors='pt', max_length=128, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.to('cpu')
    pred = label_map.get(int(torch.argmax(logits)), 'other')  # scale classes down to 5 only
    return pred