import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from onnxruntime import ExecutionMode, InferenceSession, SessionOptions

from inference import PythonInference, OnxxInference
from utils import preprocess, download_label_mapping, output_vector_to_labels, measurements_to_dataframe

def read_test_sequences(path: str):
    with open(path, 'r') as f:
        sequences = [x.rstrip() for x in f.readlines()]
        return sequences

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", torchscript=True)
    clf = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", torchscript=True)

    input_texts = [preprocess(x) for x in read_test_sequences("test_sequences.txt")]
    tokenized_inputs = [tokenizer(x, return_tensors='pt') for x in input_texts]
    
    n_experiments = 1000
    indices = np.tile(np.arange(len(input_texts)), n_experiments)

    
    # 1. Eager
    eager_model = PythonInference(clf)
    eager_measurements = eager_model.check_inference_time_all(tokenized_inputs, n_experiments)
    df_eager = measurements_to_dataframe(eager_measurements.flatten(), indices)
    df_eager['Mode'] = 'Eager'

    
    # 2. TorchScript (JIT)
    traced_model = PythonInference(model=torch.jit.trace(clf, (tokenized_inputs[0]['input_ids'], tokenized_inputs[0]['attention_mask'])))
    script_measurements = traced_model.check_inference_time_all(tokenized_inputs, n_experiments)
    df_script = measurements_to_dataframe(script_measurements.flatten(), indices)
    df_script['Mode'] = 'Script'

    
    # 3. ONNX Runtime
    model = OnxxInference(session=InferenceSession("onnx_model/twitter-roberta-base-sentiment-optimized-quantized.onnx"))
    onnx_measurements = model.check_inference_time_all(tokenized_inputs, n_experiments)
    df_onnx = measurements_to_dataframe(onnx_measurements.flatten(), indices)
    df_onnx['Mode'] = 'ONNX'


    # Statistics
    plt.style.use('seaborn')
    plt.figure()
    plt.scatter(x=df_eager['SequenceId'], y=df_eager['TimeInSeconds'], label='Eager mode')
    plt.scatter(x=df_script['SequenceId'], y=df_script['TimeInSeconds'], label='Script mode')
    plt.scatter(x=df_onnx['SequenceId'], y=df_onnx['TimeInSeconds'], label='ONNX mode')
    plt.xlabel('Sequence ID')
    plt.ylabel('Inference time [s]')
    plt.legend()
    plt.show()


    plt.figure()
    df_all = pd.concat([df_eager, df_script, df_onnx])
    df_all.groupby('Mode').mean().TimeInSeconds.plot(kind='bar')
    plt.title('Avg. inference time in seconds')
    plt.ylabel('Inference time [s]')
    plt.show()

    # Box plots