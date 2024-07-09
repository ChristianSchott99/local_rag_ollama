from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

from langchain_huggingface import HuggingFaceEmbeddings

"""
def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
"""


def get_ollama_embedding_function():
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # print("Embeddings loaded")
    # switch to german embedding model through huggingface

    embeddings = HuggingFaceEmbeddings(model="deepset/gbert-base")
    return embeddings


def get_english_embedding_function():
    """ debug
    if torch.cuda.is_available():
        print("Using GPU")
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        # use cuda gpu
        model_kwargs={'device': 'cuda:0'}
    )
    # print(f"embedding = {embeddings.model_name}")
    return embeddings


def get_german_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="deepset/gbert-base",
        # model_kwargs={'device': 'cpu'},
        # encode_kwargs={'normalize_embeddings': True}
    )
