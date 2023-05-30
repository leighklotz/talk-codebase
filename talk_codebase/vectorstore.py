"""
Module for managing vector stores for conversational AI models.

The `create_vector_store` function creates a new vector store from a set of text documents and an OpenAI API key. If there is an existing vector store, the user is prompted to use it or create a new one. The `calculate_cost` function estimates the cost of creating a new vector store based on the number of documents and the model name.


Functions:
- create_vector_store
- calculate_cost
"""

import questionary
import tiktoken
from halo import Halo
from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from talk_codebase.utils import load_files

def calculate_cost(texts, model_name):
    """
    Calculate the cost of creating a vector store for a given list of texts.

Args:
    texts: A list of texts.
    model_name: The name of the language model to use for encoding.

Returns:
    The cost of creating a vector store for the given texts.
    """
    enc = tiktoken.encoding_for_model(model_name)
    all_text = ''.join([text.page_content for text in texts])
    tokens = enc.encode(all_text)
    token_count = len(tokens)
    cost = (token_count / 1000) * 0.0004
    return cost

def get_local_vector_store(embeddings):
    try:
        return FAISS.load_local("vector_store", embeddings)
    except:
        return None

def create_vector_store(root_dir: str, openai_api_key: str, model_name: str) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    new_db = get_local_vector_store(embeddings)
    if new_db is not None:
        approve = questionary.select(
            f"Found existing vector store. Do you want to use it?",
            choices=[
                {"name": "Yes", "value": True},
                {"name": "No", "value": False},
            ]
        ).ask()
        if approve:
            return new_db

    docs = load_files(root_dir)
    if len(docs) == 0:
        print("✘ No documents found")
        exit(0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    cost = calculate_cost(docs, model_name)
    approve = questionary.select(
        f"Creating a vector store for {len(texts)} documents will cost ~${cost:.5f}. Do you want to continue?",
        choices=[
            {"name": "Yes", "value": True},
            {"name": "No", "value": False},
        ]
    ).ask()

    if not approve:
        exit(0)

    spinners = Halo(text='Creating vector store', spinner='dots').start()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("vector_store")
    spinners.succeed(f"Created vector store with {len(texts)} documents")

    return db
