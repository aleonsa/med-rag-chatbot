import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)


def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path does not exist")

        logger.info(f"Loading files from {DATA_PATH}")

        loader = DirectoryLoader(
            DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)

        documents = loader.load()

        # show how many documents were loaded
        if not documents:
            logger.warning(
                "No PDF documents found in the specified directory.")
        else:
            logger.info(f"Loaded {len(documents)} PDF documents successfully.")

        return documents

    except Exception as e:
        error_message = CustomException("Failed to load PDF files", e)
        logger.error(str(error_message))
        return []


def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents to process for text chunking")

        logger.info(f"Splitting {len(documents)} documents into text chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Created {len(text_chunks)} text chunks successfully.")

        return text_chunks

    except Exception as e:
        error_message = CustomException("Failed to create text chunks", e)
        logger.error(str(error_message))
        return []
