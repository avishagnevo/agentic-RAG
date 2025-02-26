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


with open("keys.json", "r") as config_file:
    config = json.load(config_file)

PINECONE_API_KEY = config["PINECONE_API_KEY"]
PINECONE_ENV = config["PINECONE_ENV"]
INDEX_NAME = config["INDEX_NAME"]



# Initialize Pinecone
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# index = pinecone.Index(INDEX_NAME)

class Dataset:
    """
    Handles all operations related to dataset management, including building, upserting, deleting, and retrieving data.
    """
    def __init__(self, model, data_episodes_path, data_podcasts_path):
        self.model = model
        self.data_episodes = pd.read_csv(data_episodes_path, header=0)
        self.data_podcasts = pd.read_csv(data_podcasts_path, header=0)


    def get_metadata(self, entry_id):
        """
        Prepearing metadata for the given entry
        :param entry: entry to process
        :return: metadata for entry
        """
        #
        metadata = {}

        if entry_id.startswith('p'):
            entry=self.data_podcasts[self.data_podcasts['id'] == entry_id].iloc[0]            
            metadata['dataset'] = 'podcasts'
            metadata['title'] = entry['title']
            metadata['description'] = entry['description']    
            metadata['rating'] = float(entry['average_rating'])
            metadata['category'] = entry['category']
            metadata['itunes_url'] = entry['itunes_url']
            #itunes_url,title,description,average_rating,category,id,to_embed
        
        elif entry_id.startswith('e'):
            entry=self.data_episodes[self.data_episodes['id'] == entry_id].iloc[0]
            metadata['dataset'] = 'episodes'
            metadata['episode_name'] = entry['episodeName']
            metadata['show_name'] = entry['show.name']
            metadata['episode_description'] = entry['description']
            metadata['show_description'] = entry['show.description']
            metadata['duration_min'] = int(entry['duration_ms'])
            #id,episodeUri,showUri,episodeName,description,show.name,show.description,show.publisher,duration_ms,to_embed

        else:
            raise ValueError(f"Invalid dataset. Choose from 'episodes' or 'podcasts'")    

        print(metadata)

        return metadata    


class Index:
    def __init__(self, model, dataset, index_name: str, dimension: int, metric = "cosine"):
        self.model = model
        self.dataset = dataset
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.pc = self.init_client()
        self.index = self.pc.Index(self.index_name)

    def init_client(self):
        """
        Initialize Pinecone client
        :param api_key:
        :return:
        """
        pc = Pinecone(api_key=PINECONE_API_KEY)
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
        self.pc.create_index(
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
        entries_ids = entries['id']
        entries_text = entries['to_embed']

        for entry_id, embedding, text in zip(entries_ids, embeddings, entries_text):
            metadata = self.dataset.get_metadata(entry_id)
            vectors.append({
                "id": str(entry_id),
                "values": embedding,
                "metadata": {
                    "text": text,
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

        for i, j in zip(range(0, len_data, 30), range(30, len_data+1, 30)):
            data = dataset[i:j]  # For testing purposes, only use a subset of the data

            # Embedding model to generate embeddings for the texts
            data_to_embed = [d for d in data['to_embed']]
            embeddings = self.model.get_docs_embedding(data_to_embed)
            print('embedding dim:', len(embeddings[0]))

            self.add_to_index(data, embeddings)
            # Sleep for 0.5 second
            time.sleep(0.2)    

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


def init_database_with_upsert():
    index_name = "agentic-rag"
    dimension = 1024

    # Load the data
    data_episodes_path = 'data/episodes.csv'
    data_podcasts_path = 'data/podcasts.csv'

    model = AzureOpenAIModels()
    dataset = Dataset(model, data_episodes_path, data_podcasts_path)

    index = Index(model, dataset, index_name, dimension, metric="cosine")

    # Uncomment to create the index (only needs to be done once)
    # index.create_index()

    index.upsert_by_chunks(dataset.data_episodes)
    index.upsert_by_chunks(dataset.data_podcasts)

    return index, dataset, model


def init_database():
    index_name = "agentic-rag"
    dimension = 1024

    # Load the data
    data_episodes_path = 'data/episodes.csv'
    data_podcasts_path = 'data/podcasts.csv'

    model = AzureOpenAIModels()
    dataset = Dataset(model, data_episodes_path, data_podcasts_path)

    index = Index(model, dataset, index_name, dimension, metric="cosine")

    return index, dataset, model


if __name__ == "__main__":
    init_database_with_upsert()
        