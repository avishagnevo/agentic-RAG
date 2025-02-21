import os
import pandas as pd
#import pinecone
import json
from openai import AzureOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from llms import AzureOpenAIModels



# Replace these placeholders with your actual Azure OpenAI credentials
#AZURE_OPENAI_API_KEY = "your-azure-openai-key"
#AZURE_ENDPOINT = "your-azure-endpoint"
#API_VERSION = "2023-05-15"
#DEPLOYMENT_NAME = "your-deployment-name"
#PINECONE_API_KEY = "your-pinecone-api-key"
#PINECONE_ENV = "your-pinecone-environment"
#INDEX_NAME = "podcast-search"


class Agent:
    """
    Represents an AI agent that performs micro tasks.
    """
    
    def __init__(self, name, model, system_prompt_template, prompt_template):
        # Load the JSON file
        with open("agent_templates.json", "r") as file:
            agents_data = json.load(file)

        # Print to verify
        self.AGENT_NAMES = agents_data.keys()  # Should print all agent names

        if name not in self.AGENT_NAMES:
            raise ValueError(f"Invalid agent type. Choose from {AGENT_NAMES}")

        self.SYSTEM_PROMPT_TEMPLATES = agents_data[name]["SYSTEM_PROMPT_TEMPLATES"]

        if system_prompt_template not in self.SYSTEM_PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template. Choose from {self.SYSTEM_PROMPT_TEMPLATES}")                

        self.PROMPT_TEMPLATES = agents_data[name]["PROMPT_TEMPLATES"]

        if prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template. Choose from {self.PROMPT_TEMPLATES}")    

        self.name = name
        self.model = AzureOpenAIModels().chat_model
        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template
        self.prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATES[prompt_type])
        self.instruction_prompt = agents_data[name]["PROMPT_TEMPLATES"][prompt_template]

    def run(self, input_data):
        """Executes the agent with the given input and returns structured output."""
        formatted_prompt = self.instruction_prompt.format(input=input_data) #get the user input into a chosen tamplate
        agent_output = self.model.get_chat_response(self.system_content, formatted_prompt)

        return agent_output
