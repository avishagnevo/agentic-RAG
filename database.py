import os
import time
import os
import pandas as pd
import pinecone
import json
import itertools
from openai import AzureOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
from llms import AzureOpenAIModels


# Replace these placeholders with your actual Azure OpenAI credentials
AZURE_OPENAI_API_KEY = "your-azure-openai-key"
AZURE_ENDPOINT = "your-azure-endpoint"
API_VERSION = "2023-05-15"
DEPLOYMENT_NAME = "your-deployment-name"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "your-pinecone-environment"
INDEX_NAME = "podcast-search"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

class Dataset:
    """
    Handles all operations related to dataset management, including building, upserting, deleting, and retrieving data.
    """
    def __init__(self, model, data_episodes_path, data_podcasts_path):
        self.model = model
        self.data_episodes = pd.read_csv(data_episodes_path, ignore_index=True)
        self.data_podcasts = pd.read_csv(data_podcasts_path, ignore_index=True)

    
    def get_metadata(self, entry):
        """
        Prepearing metadata for the given entry
        :param entry: entry to process
        :return: metadata for entry
        """
        #
        metadata = {}
        
        if entry['dataset'] == 'episodes':
            metadata['dataset'] = 'episodes'
            metadata['episode_name'] = entry['episodeName']
            metadata['show_name'] = entry['show.name']
            metadata['episode_description'] = entry['description']
            metadata['show_description'] = entry['show.description']
            metadata['duration_ms'] = entry['duration_ms']
            metadata['id'] = entry['id']
            #id,episodeUri,showUri,episodeName,description,show.name,show.description,show.publisher,duration_ms,to_embed

        elif entry['dataset'] == 'podcasts':
            metadata['dataset'] = 'podcasts'
            metadata['title'] = entry['title']
            metadata['description'] = entry['description']    
            metadata['id'] = entry['id']
            metadata['rating'] = entry['average_rating']
            metadata['category'] = entry['category']
            metadata['itunes_url'] = entry['itunes_url']
            #itunes_url,title,description,average_rating,category,id,to_embed

        else:
            raise ValueError(f"Invalid dataset. Choose from 'episodes' or 'podcasts'")    

        return metadata    


class Index:
    def __init__(self, embedding_model, dataset, api_key, index_name: str, dimension: int, metric = "cosine"):
        self.embedding_model = embedding_model
        self.dataset = dataset
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.api_key = api_key
        self.index = self.init_client()

    def init_client(self):
        """
        Initialize Pinecone client
        :param api_key:
        :return:
        """
        PINECONE_API_KEY = self.api_key
        index = Pinecone(api_key=PINECONE_API_KEY)
        return pc


    def create_index(self):
        """
        Create a Pinecone index, if doesn't exist
        :param pc: pinecone object
        :param index_name:
        :param dimension:
        :param metric:
        :return:
        """
        self.index.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )


    def add_to_index(self, entries, embeddings):
        """
        Adding cleaned metadata to index, then updating and inserting.
        :param api_key:
        :param entries:
        :param embeddings:
        :param index_name:
        :return:
        """
        #pc = Pinecone(api_key=api_key)
        #index = pc.Index(index_name)

        vectors = []
        for entry, embedding in zip(entries, embeddings):
            metadata = self.dataset.get_metadata(entry)
            vectors.append({
                "id": str(entry['id']),
                "values": embedding,
                "metadata": {
                    "text": entry['to_embed'],
                    **metadata
                }
            })
        self.index.upsert(vectors=vectors, namespace="ns0")

    def upsert_by_chunks(self, dataset):
        """
        Upsert data in chunks
        :param data:
        :param embeddings:
        :return:
        """
        len_data = len(dataset)

        for i, j in zip(range(30, len_data, 30), range(60, len_data, 30)):
            data = dataset[i:j]  # For testing purposes, only use a subset of the data

            # Embedding model to generate embeddings for the texts
            data_to_embed = [d['to_embed'] for d in data]
            embeddings = self.embedding_model.get_docs_embedding(data_to_embed)
            print('embedding dim:', len(embeddings[0]))

            # Upsert in batches
            for data_chunk, embeddings_chunk in zip(chunks(data, 1), chunks(embeddings, 1)):
                add_to_index(data_chunk, embeddings_chunk)
                # Sleep for 0.5 second
                time.sleep(0.5)

            print(f"Dataset[{i}:{j}] added to the Pinecone index!")        
         


    def chunks(self, iterable, batch_size=200):
        """
        Helper function to break an iterable object into chunks
        :param iterable: iterable object
        :param batch_size: size of batch
        :return:
        """
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))


    def retrieve_data(self, query_embedding, top_k=5, filters=None):
        """Performs semantic search with metadata filtering."""
        filter_query = filters if filters else {}
        results = self.index.query(vector=query_embedding, top_k=top_k, filter=filter_query, include_metadata=True)
        return results["matches"]        



def init_database():
    PINECONE_API_KEY = #'bb68c35d-a2f2-47a7-9d21-78f0e3b0ab68'
    index_name = "agentic-RAG" 
    dimension = #0

    # Load the data
    data_episodes_path = 'data/episodes.csv'
    data_podcasts_path = 'data/podcasts.csv'

    model = AzureOpenAIModels()
    embedding_model = model.embedding_model
    dataset = Dataset(embedding_model, data_episodes_path, data_podcasts_path)

    index = Index(embedding_model, dataset, PINECONE_API_KEY, index_name, dimension, metric="cosine")

    # Uncomment to create the index (only needs to be done once)
    # index.create_index()

    index.upsert_by_chunks(dataset.data_episodes)
    index.upsert_by_chunks(dataset.data_podcasts)

    return index, dataset, model


if __name__ == "__main__":
    init_database()
        