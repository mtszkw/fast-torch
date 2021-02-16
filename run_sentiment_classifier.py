import csv
import urllib.request

import torch
from transformers import AutoTokenizer

from preprocessing import preprocess
# from tokenizer import PretrainedTokenizer
from classifier import PretrainedSequenceClassifier

def download_label_mapping():
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    return labels

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", torchscript=True)
# tokenizer = PretrainedTokenizer("cardiffnlp/twitter-roberta-base-sentiment")

clf = PretrainedSequenceClassifier("cardiffnlp/twitter-roberta-base-sentiment", labels=download_label_mapping())

input_text = preprocess("I don't think it's gonna work")

tokenized_input = tokenizer(input_text, return_tensors='pt')
print(tokenized_input)

# 1. Vanilla
output = clf.predict_labels(tokenized_input)
print(output)

# 2. TorchScript (JIT)
# encoded_input = tokenizer.encode(input_text)
# print(encoded_input)

# print(clf.predict_labels(tokenized_input))

# input_ids = tokenizer.tokenizer.convert_tokens_to_ids(encoded_input)
# print(indexed_tokens)
# input_ids = torch.tensor(encoded_input).unsqueeze(0)
# print(input_ids)

# print(input_ids)
# traced_model = torch.jit.trace(clf.model, input_ids)
# print(traced_model)
# torch.jit.save(traced_model, "traced_twitter_roberta_base_sentiment.pt")