import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import sys

sys.path.append('.')

from src.data import TelegramData


def vectorize_corpus(chunks) -> np.ndarray:
    MODEL_NAME = "cointegrated/LaBSE-en-ru"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    sentenses = chunks

    embeddings_list = []

    for s in sentenses:
        encoded_input = tokenizer(s, padding=True, truncation=True, max_length=64, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        embedding = model_output.pooler_output
        embeddings_list.append((embedding)[0].numpy())

    embeddings = np.asarray(embeddings_list)
    return embeddings

