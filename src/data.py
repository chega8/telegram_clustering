from typing import List, Optional, Any
import re
import pandas as pd
import torch
import numpy as np
from abc import ABC, abstractmethod

from src.utils import clean, split_string_into_chunks


class BaseData(ABC):
    def __init__(self, source: Any = None) -> None:
        self.source = source

    @abstractmethod
    def get_chunks(self) -> List[str]:
        ...


class VectorizedData:
    def __init__(
        self, raw_data: BaseData, embeddings: Optional[torch.Tensor] = None
    ) -> None:
        self.raw_data = raw_data
        self.embeddings = embeddings

    def get_chunks(self) -> List[str]:
        return self.raw_data.get_chunks()

    def get_embeddings(self):
        return self.embeddings

    def save_embeddings(self, pth: str = "data/embeddings.pt"):
        if self.embeddings is not None:
            torch.save(self.embeddings, pth)

    def load_embeddings(self, pth: str = "data/embeddings.pt"):
        self.embeddings = torch.load(pth)


class ChannelData(BaseData):
    ...


class ChatData(BaseData):
    def __init__(self, source: Any = None) -> None:
        self.source = "data/chat_history.txt"
        with open(self.source, "r") as f:
            self.data = f.read()

        self.data = clean(self.data)

    def get_chunks(self) -> List[str]:
        return split_string_into_chunks(self.data, 300)
