from telethon.sync import TelegramClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

def scrap_history(chat_id):
    out_file = chat_id.split('/')[-1]
    with TelegramClient('anon', api_id, api_hash) as client:
        with open(f'data/{out_file}.txt', 'w', encoding='utf-8') as f:
            for message in client.iter_messages(chat_id):
                f.write(f"{message.sender_id}: {message.text}\n")
            
            
if __name__ == '__main__':
    chat_id = ''
    scrap_history(chat_id)