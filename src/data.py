from typing import List
import re
import pandas as pd
import numpy as np


def remove_links(text):
    # This regex matches most common types of URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)

def clean(text):    
    text = remove_links(text)

    lines = text.split("\n")[::-1]
    text = ' '.join(lines)
    
    ids = set(['1188103941:', '566572635:'])

    words = text.split(" ")
    words = [w for w in words if w != '' and w != ' ']
    messages = []
    segment = []
    for w in words:
        if w in ids:
            messages.append(' '.join(segment))
            segment = []
            
        segment.append(w)
            
    messages = [msg for msg in messages if len(msg) < 100]
    text = ' '.join(messages)
    
    for i in ids:
        text = text.replace(i, '')
    text = text.replace('None', '')
    # text = text.replace('  ', '')
    return text

def split_string_into_chunks(s, num_chunks):
    # Calculate the length of the string
    str_len = len(s)
    
    # Determine the approximate size of each chunk
    avg_chunk_size = str_len // num_chunks

    # Create the chunks
    chunks = [s[i:i + avg_chunk_size] for i in range(0, str_len, avg_chunk_size)]

    return chunks

def split_df_into_chunks(df, num_chunks):
    chunks = np.array_split(df, num_chunks)
    return [' '.join(chunk['msg'].values) for chunk in chunks]


class TelegramData:
    def __init__(self, source) -> None:
        self.source = source
        
    def get_chunk(self) -> str:
        ...
        
    def get_chunks(self) -> List[str]:
        ...
        

class ChannelData(TelegramData):
    ...
    

class ChatData(TelegramData):
    def __init__(self, source) -> None:
        self.source = 'data/chat_history.txt'
        with open(self.source, 'r') as f:
            self.data = f.read()
            
        self.data = clean(self.data)
        
    def get_chunks(self) -> List[str]:
        return split_string_into_chunks(self.data, 300)
    