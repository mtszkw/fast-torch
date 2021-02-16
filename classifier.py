import torch
import numpy as np
from scipy.special import softmax

from transformers import AutoModelForSequenceClassification

class PretrainedSequenceClassifier:
    def __init__(self, model_name: str, labels):
        self.model_name = model_name
        self.labels = labels
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torchscript=True)
        # self.model.save_pretrained(self.model_name)

    def predict(self, encoded_input):
        # text = preprocess(text)
        # encoded_input = self.tokenizer(text, return_tensors='pt')
        self.model.eval()
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores

    def predict_labels(self, encoded_input):
        scores = self.predict(encoded_input)

        scores_map = {}
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = self.labels[ranking[i]]
            s = scores[ranking[i]]
            scores_map[l] = np.round(float(s), 4)
            # print(f"{i+1}) {l} {np.round(float(s), 4)}")
        return scores_map

