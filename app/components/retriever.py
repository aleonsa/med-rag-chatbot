from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import save_vector_store, load_vector_store

from app.config.config import OPENAI_API_KEY, OPENAI_MODEL_NAME

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
You are a medical chatbot. You will answer questions in 2-3 lines maximum using only the information provided in the context.

Context: {context}

Question: {question}

Answer:
"""


def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException(
                "Vector store not found. Please create a new one.")

        llm = load_llm(openai_api_key=OPENAI_API_KEY,
                       openai_model_name=OPENAI_MODEL_NAME)

        if llm is None:
            raise CustomException(
                "LLM not loaded. Please check your configuration.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": set_custom_prompt()}
        )

        logger.info("QA chain created successfully.")

        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to create QA chain", e)
        logger.error(str(error_message))
        raise
