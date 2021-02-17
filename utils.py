import csv
import numpy as np
import urllib.request
from scipy.special import softmax

def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def download_label_mapping():
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    return labels

def output_vector_to_labels(output, labels_map):
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    scores_map = {}
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels_map[ranking[i]]
        s = scores[ranking[i]]
        scores_map[l] = np.round(float(s), 4)
        # print(f"{i+1}) {l} {np.round(float(s), 4)}")
    return scores_map