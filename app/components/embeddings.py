from langchain_openai import OpenAIEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import OPENAI_API_KEY

import os

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Initializing OpenAI embeddings model...")
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY no encontrada")

        model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        logger.info("OpenAI embeddings model initialized successfully.")
        return model
    except Exception as e:
        error_message = CustomException(
            "Failed to initialize OpenAI embeddings model", e)
        logger.error(str(error_message))
        raise error_message
