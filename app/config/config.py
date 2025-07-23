from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINFACEHUB_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4.1-nano"


DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
