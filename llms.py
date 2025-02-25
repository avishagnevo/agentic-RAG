import os
import json
import pandas as pd
import pinecone
from openai import AzureOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory


'''
o3-mini version 2025-01-31
o1 version: 2024-12-17
gpt-4o-mini version: 2024-07-18
gpt-4o version: 2024-08-06
gpt-4o version: 2024-11-20
'''


# Open and load the configuration file
with open("keys.json", "r") as config_file:
    config = json.load(config_file)

AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_ENDPOINT = config["AZURE_ENDPOINT"]
API_VERSION = config["API_VERSION"]
CHAT_DEPLOYMENT = config["CHAT_DEPLOYMENT"]
EMBEDDING_DEPLOYMENT = config["EMBEDDING_DEPLOYMENT"]


class AzureOpenAIModels:
    """
    Handles all interactions with the Azure OpenAI API, including embedding generation and chat model usage.
    """
    def __init__(self, model_name = "gpt-4o"):
        self.chat_model = AzureChatOpenAI(
            azure_deployment=CHAT_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION,
            openai_api_type="azure",
            temperature=0.5,
            model=model_name,
            # max_tokens=None,
            # timeout=None,
            # max_retries=2,
            # model_version="0125",
        )

        self.embedding_model = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_deployment=EMBEDDING_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            openai_api_version="2023-05-15",
            openai_api_type="azure",
            dimensions = 1024, # Can specify dimensions with new text-embedding-3 models
        )

    def get_docs_embedding(self, docs : list[str]) -> list[list[float]]:
        """Generates an embedding for the given list of texts using Azure OpenAI."""
        docs_embeddings = self.embedding_model.embed_documents(docs)
        print(docs_embeddings[0][:10])  # Show the first 10 characters of the first vector

        return docs_embeddings

    def get_query_embedding(self, query : str) -> list[float]:
        """Generates an embedding for the given text using Azure OpenAI."""
        query_embedding = self.embedding_model.embed_query(query)

        return query_embedding

    def get_chat_response(self, system_prompt : str, prompt : str) -> str:
        """Generates a response using Azure OpenAI chat model."""

        messages = [
            ("system", system_prompt),
            ("human", prompt),
        ]
        raw_respond = self.chat_model.invoke(messages)
        return raw_respond.content