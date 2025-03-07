import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import SimpleSequentialChain


# Config values
load_dotenv() # Load environment variables from .env file
AZURE_OPENAI_API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2024-02-01"
CHAT_DEPLOYMENT = "team2-gpt4o"
EMBEDDING_DEPLOYMENT = "team2-embedding"

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
        )

        self.embedding_model = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_deployment=EMBEDDING_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            openai_api_version=API_VERSION,
            openai_api_type="azure",
            dimensions = 1024, # 1024 for text-embedding-3-large
        )

    def get_docs_embedding(self, docs : list[str]) -> list[list[float]]:
        """Generates an embedding for the given list of texts using Azure OpenAI."""
        docs_embeddings = self.embedding_model.embed_documents(docs)
        
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
