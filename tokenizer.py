from transformers import AutoTokenizer

from preprocessing import preprocess

class PretrainedTokenizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torchscript=True)

    def __call__(self, text: str):
        text = preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        return encoded_input
