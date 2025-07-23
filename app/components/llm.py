from langchain_openai import ChatOpenAI
from app.config.config import OPENAI_API_KEY, OPENAI_MODEL_NAME

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


def load_llm(openai_api_key: str = OPENAI_API_KEY, openai_model_name: str = OPENAI_MODEL_NAME):
    try:
        logger.info(f"Loading LLM from OpenAI: {openai_model_name}")

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=openai_model_name,
            temperature=0.35,
            max_tokens=256,
        )

        logger.info("OpenAI LLM loaded successfully.")

        return llm

    except Exception as e:
        error_message = CustomException("Failed to load OpenAI LLM", e)
        logger.error(str(error_message))
        raise error_message