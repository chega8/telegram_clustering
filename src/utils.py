from typing import List, Optional, Any
import re
import numpy as np
import pandas as pd


def remove_links(text: str) -> str:
    # This regex matches most common types of URLs
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return url_pattern.sub("", text)


def clean(text: str) -> str:
    text = remove_links(text)

    lines = text.split("\n")[::-1]
    text = " ".join(lines)

    ids = set(["1188103941:", "566572635:"])

    words = text.split(" ")
    words = [w for w in words if w != "" and w != " "]
    messages = []
    segment = []
    for w in words:
        if w in ids:
            messages.append(" ".join(segment))
            segment = []

        segment.append(w)

    messages = [msg for msg in messages if len(msg) < 100]
    text = " ".join(messages)

    for i in ids:
        text = text.replace(i, "")
    text = text.replace("None", "")
    return text


def split_string_into_chunks(s: str, num_chunks: int) -> List[str]:
    str_len = len(s)

    avg_chunk_size = str_len // num_chunks

    chunks = [s[i : i + avg_chunk_size] for i in range(0, str_len, avg_chunk_size)]

    return chunks


def split_df_into_chunks(df: pd.DataFrame, num_chunks: int) -> List[str]:
    chunks = np.array_split(df, num_chunks)
    return [" ".join(chunk["msg"].values) for chunk in chunks]
