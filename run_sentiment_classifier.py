# https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
from timeit import default_timer as timer

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from utils import preprocess, download_label_mapping, output_vector_to_labels


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

    input_texts = [
        "Hello world",
        "Happy birthday",
        "I don't think it's gonna work",
        "I enjoy natural language understanding"
    ]

    # 1. Vanilla
    tokenized_inputs = [tokenizer(x, return_tensors='pt') for x in input_texts]
    outputs = [run_model(clf, x) for x in tokenized_inputs]
    output_times = [check_inference_time(clf, x) for x in tokenized_inputs]
    
    for inp, out in zip(input_texts, outputs):
        print(inp, out)

    print(output_times)

    print("")

    # 2. TorchScript (JIT)
    tokenized_inputs = [tokenizer(x, return_tensors='pt') for x in input_texts]
    traced_model = torch.jit.trace(clf, (tokenized_inputs[0]['input_ids'], tokenized_inputs[0]['attention_mask']))
    outputs = [run_model(traced_model, x) for x in tokenized_inputs]
    output_times = [check_inference_time(traced_model, x) for x in tokenized_inputs]
    # torch.jit.save(traced_model, "traced_twitter_roberta_base_sentiment.pt")
    # loaded_model = torch.jit.load("traced_twitter_roberta_base_sentiment.pt")
    for inp, out in zip(input_texts, outputs):
        print(inp, out)

    print(output_times)
