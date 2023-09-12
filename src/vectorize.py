import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import sys

sys.path.append(".")

from src.data import BaseData, VectorizedData


MODEL_NAME = "cointegrated/LaBSE-en-ru"


def vectorize_chunks(chunks: List[str]) -> torch.Tensor:
    """Vectorize chunk of sentences"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    embeddings_list = []
    for sentence in chunks:
        encoded_input = tokenizer(
            sentence, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = model(**encoded_input)

        embedding = model_output.pooler_output
        embeddings_list.append((embedding)[0].numpy())

    embeddings = np.asarray(embeddings_list)
    return embeddings


def get_vectorized_data(raw_data: BaseData) -> VectorizedData:
    embeddings = vectorize_chunks(raw_data.get_chunks())

    return VectorizedData(raw_data, embeddings)
