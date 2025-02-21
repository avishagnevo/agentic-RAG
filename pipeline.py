import os
import json
import pandas as pd
import pinecone
from openai import AzureOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

from agent import Agent
import database
from llms import AzureOpenAIModels

# Replace these placeholders with your actual Azure OpenAI credentials
AZURE_OPENAI_API_KEY = "your-azure-openai-key"
AZURE_ENDPOINT = "your-azure-endpoint"
API_VERSION = "2023-05-15"
DEPLOYMENT_NAME = "your-deployment-name"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "your-pinecone-environment"
INDEX_NAME = "podcast-search"


class AgenticPipeline:
    """
    Manages the structured execution of AI agents to process user queries and retrieve relevant podcast recommendations.
    """
    def __init__(self, index, dataset, model):
        self.index = index
        self.dataset = dataset
        self.model = model
        self._initialize_agents()

    def _initialize_agents(self):
        """Initializes all agents using the pre-defined templates from agent_templates.json."""
        with open("agent_templates.json", "r") as file:
            agents_data = json.load(file)
        
        self.agents = {
            "QueryInitialCheck": Agent("QueryInitialCheck", self.model, "structured", "QuestionRefinementPattern"), # or "InstructionBased"
            #"UserProcessing": Agent("UserProcessing", self.model, "structured", "InstructionBased"), #is redundent, Adi also say that
            "SearchFilters": Agent("SearchFilters", self.model, "structured", "FewShot"),
            "NeedUnderstanding": Agent("NeedUnderstanding", self.model, "structured", "PersonaPattern"),
            "ResponseGeneration": Agent("ResponseGeneration", self.model, "unstructured", "AudiencePattern"),
            "Supervision": Agent("Supervision", self.model, "structured", "FewShot"),
        }
    
    def execute(self, user_query):
        """
        Runs the full agentic pipeline, ensuring each step feeds into the next logically.
        """
        print("Starting Agentic Pipeline Execution...")

        # Step 0: Database Selection
        query_pass = self.agents["QueryInitialCheck"].run(user_query) #should return an output like a Pinecone filter
        print("User Query Pass?", query_pass)

        if not query_pass: #not luke that, its just pseodu code
            return "Sorry, I didn't understand your query. Please try again." #or something else generated with "QueryInitialCheck" agent

        # Step 1: User Input Processing
        processed_input = # Implement a function, should be handled without an agant, but in case self.agents["UserProcessing"].run(user_query)
        print("Processed Input:", processed_input)
        
        # Step 2: Index Filters Extraction
        search_filters = self.agents["SearchFilters"].run(processed_input) #should return an output like a Pinecone filter
        print("Search Details:", search_details)
        
        # Step 3: Need Understanding & Augmentation
        needs_summary = self.agents["NeedUnderstanding"].run(processed_input)
        print("Needs Summary:", needs_summary)
        
        # Step 4: Semantic Search using Pinecone
        query_embedding = self.model.get_query_embedding(needs_summary)
        search_results = self.index.retrieve_data(query_embedding, top_k=5, filters=search_filters)
        print("Semantic Search Results:", search_results)

        #maybe add another supervision here?
        
        # Step 5: Augmented Prompt Construction
        augmented_prompt = {
            "search_results": search_results,
            "needs_summary": needs_summary
            "user_query": user_query
        }
        print("Augmented Prompt:", augmented_prompt)
        
        # Step 6: Response Generation
        response = self.agents["ResponseGeneration"].run(augmented_prompt)
        print("Generated Response:", response)
        
        # Step 7: Supervision & Refinement
        final_response = self.agents["Supervision"].run(response)
        print("Final Validated Response:", final_response)

        #maybe only if PASS then return final_response, otherwise do something else
        
        return final_response

def initialize_index():
    """
    Initialize the Pinecone index for the podcast dataset.
    """
         

if __name__ == "__main__":
    # Define dataset paths
    data_episodes_path = "data/episodes.csv"
    data_podcasts_path = "data/podcasts.csv"

    index, dataset, embedding_model = database.init_database()
    
    # Initialize and execute the pipeline
    pipeline = AgenticPipeline(data_episodes_path, data_podcasts_path)
    user_prompt = "Find me top Data Science podcasts."
    final_output = pipeline.execute(user_prompt)
    print("Final Output:", final_output)
