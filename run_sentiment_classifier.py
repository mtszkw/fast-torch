import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
from timeit import default_timer as timer

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from utils import preprocess, download_label_mapping, output_vector_to_labels


def read_test_sequences(path: str):
    with open(path, 'r') as f:
        sequences = [x.rstrip() for x in f.readlines()]
        return sequences


def run_model(model, tokenized_input):
    output = model(**tokenized_input)
    return output_vector_to_labels(output, download_label_mapping())


def check_inference_time(model, tokenized_input):
    t = timer()
    scores = run_model(model, tokenized_input)
    elapsed_time = timer()-t
    return elapsed_time


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", torchscript=True)
    clf = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", torchscript=True)

    n_experiments = 5
    input_texts = [preprocess(x) for x in read_test_sequences("test_sequences.txt")]

    # 1. Eager
    eager_measurements = np.zeros((n_experiments, len(input_texts)))
    tokenized_inputs = [tokenizer(x, return_tensors='pt') for x in input_texts]

    for i in range(n_experiments):
        # outputs = [run_model(clf, x) for x in tokenized_inputs]
        eager_measurements[i] = [check_inference_time(clf, x) for x in tokenized_inputs]
        # for inp, out in zip(input_texts, outputs):
            # print(inp, '\n', out, '\n')
        # print(output_times)

    
    # 2. TorchScript (JIT)
    script_measurements = np.zeros((n_experiments, len(input_texts)))
    tokenized_inputs = [tokenizer(x, return_tensors='pt') for x in input_texts]
    traced_model = torch.jit.trace(clf, (tokenized_inputs[0]['input_ids'], tokenized_inputs[0]['attention_mask']))
    # torch.jit.save(traced_model, "traced_twitter_roberta_base_sentiment.pt")
    # loaded_model = torch.jit.load("traced_twitter_roberta_base_sentiment.pt")

    for i in range(n_experiments):
        # outputs = [run_model(traced_model, x) for x in tokenized_inputs]
        script_measurements[i] = [check_inference_time(traced_model, x) for x in tokenized_inputs]
        # for inp, out in zip(input_texts, outputs):
            # print(inp, '\n', out, '\n')
        # print(output_times)

    print(eager_measurements)
    print(script_measurements)

    # Box Plot
    
    eager_avgs = np.mean(eager_measurements, axis=0)
    script_avgs = np.mean(script_measurements, axis=0)
    print(eager_avgs)
    print(script_avgs)

    # Scatter Plot

    indices = np.tile(np.arange(len(input_texts)), n_experiments)
    eager_measurements = eager_measurements.flatten()
    script_measurements = script_measurements.flatten()
    print(indices)
    print(eager_measurements)

    plt.style.use('seaborn')
    plt.scatter(indices, eager_measurements, label='Eager mode')
    plt.scatter(indices, script_measurements, label='Script mode')
    plt.xlabel('Sequence ID')
    plt.ylabel('Inference time [s]')
    plt.legend()
    plt.show()
