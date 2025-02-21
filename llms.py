import os
import pandas as pd
import pinecone
from openai import AzureOpenAI
from langchain.vectorstores import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

#%pip install -qU langchain-openai

'''
o3-mini version 2025-01-31
o1 version: 2024-12-17
gpt-4o-mini version: 2024-07-18
gpt-4o version: 2024-08-06
gpt-4o version: 2024-11-20
'''

# Replace these placeholders with your actual Azure OpenAI credentials
AZURE_OPENAI_API_KEY = "your-azure-openai-key"
AZURE_ENDPOINT = "your-azure-endpoint"
API_VERSION = "your-api-version"
DEPLOYMENT_NAME = "your-deployment-name"
#PINECONE_API_KEY = "your-pinecone-api-key"
#PINECONE_ENV = "your-pinecone-environment"
#INDEX_NAME = "podcast-search"

class AzureOpenAIModels:
    """
    Handles all interactions with the Azure OpenAI API, including embedding generation and chat model usage.
    """
    def __init__(self):
        self.chat_model = AzureChatOpenAI(
            azure_deployment=DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION,
            openai_api_type="azure"
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # model="gpt-35-turbo",
            # model_version="0125",
        )

        self.embedding_model = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_deployment=DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            openai_api_version=API_VERSION,
            openai_api_type="azure"
            # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
        )
    
    def get_docs_embedding(self, docs : List[str]) -> List[List[float]]:
        """Generates an embedding for the given list of texts using Azure OpenAI."""
        docs_embeddings = self.embedding_model.embed_documents(docs)
        print(docs_embeddings[0][:10])  # Show the first 10 characters of the first vector

        return docs_embeddings

    def get_query_embedding(self, quary : str) -> List[float]:
        """Generates an embedding for the given text using Azure OpenAI."""
        query_embedding = self.embedding_model.embed_query(query)
        print(query_embedding[:10]) # Show the first 10 characters of the first vector

        return query_embedding   
    
    def get_chat_response(self, system_prompt : str, prompt : str) -> str:
        """Generates a response using Azure OpenAI chat model."""

        messages = [
            ("system", system_prompt),
            ("human", prompt),
        ]
        raw_respond = self.chat_model.invoke(messages)     
        print(raw_respond.content)
        return raw_respond.content