# Fast Torch

I wanted to explore different ways to optimize PyTorch models for inference, so I played a little bit with TorchScript, ONNX Runtime and classic PyTorch eager-mode to compare their performance. I use pre-trained RoBERTa model (trained for sentiment analysis from tweets) along with BERT tokenizer. Both models are hosted by HuggingFace.

I wrote 14 different text sequences (7 with positive and 7 with negative sentiments) with different lengths and I used them for model inference. To obtain more reliable results, I repeated that process 1000 times (1000 times x 14 sequences = 14K runs for single model configuration).

### Results

Horizontal axis shows 14 sequences (numered from 0 to 13) that were used for prediction. In each column (for each sequence) there is n=1000 measurements for each mode: Eager, Script (JIT), ONNX. A total of 3000 values is present for each sequence ID. Eager (default) mode is always slightly worse than Script (TorchScript) mode inference. ONNX Runtime seems to outperform both Eager and Script predictions speed which can be observed in the image below.

![](doc/scatter.png)

When summing up all the results (from all experiments and sequences), grouping them by inference mode and calculating the average, it is once again clear that ONNX performs much better than other two options. 

![](doc/bar.png)