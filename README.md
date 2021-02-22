# Fast Torch

I wanted to explore different ways to optimize PyTorch models for inference, so I played a little bit with TorchScript, ONNX Runtime and classic PyTorch eager-mode and compared their performance. I use pre-trained RoBERTa model (trained for sentiment analysis from tweets) along with BERT tokenizer. Both models are [available here](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).

I wrote 14 short-to-medium length text sequences (7 with positive and 7 with negative sentiments) and I used them for model prediction. To obtain more reliable results, I repeated that process 1000 times (1000 times x 14 sequences = 14K runs for a single model configuration).

### Results

Horizontal axis shows 14 sequences (numbered from 0 to 13) that were used for prediction. In each column (for each sequence) there is n=1000 measurements for: Eager, Script (JIT), ONNX modes. A total of 3000 values were plotted for each sequence ID. Eager (default) mode is always slightly worse than Script (TorchScript) mode inference. ONNX Runtime seems to outperform both Eager and Script predictions speed which can be observed in the image below.

![](doc/scatter.png)

When summing up all the results (from all experiments and sequences), grouping them by inference mode and calculating the average, it is once again clear that ONNX performs much better than other two options. 

![](doc/bar.png)
