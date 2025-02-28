import json
from langchain.prompts import ChatPromptTemplate
from llms import AzureOpenAIModels

class Agent:
    """
    Represents an AI agent that performs micro tasks.
    """
    
    def __init__(self, name, system_prompt_template, prompt_template):
        # Load the JSON file
        with open("agent_templates.json", "r") as file:
            agents_data = json.load(file)

        self.AGENT_NAMES = agents_data.keys()

        if name not in self.AGENT_NAMES:
            raise ValueError(f"Invalid agent type. Choose from {self.AGENT_NAMES}")

        self.SYSTEM_PROMPT_TEMPLATES = agents_data[name]["SYSTEM_PROMPT_TEMPLATES"]

        if system_prompt_template not in self.SYSTEM_PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template. Choose from {self.SYSTEM_PROMPT_TEMPLATES}")                

        self.PROMPT_TEMPLATES = agents_data[name]["PROMPT_TEMPLATES"]

        if prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template. Choose from {self.PROMPT_TEMPLATES}")    

        self.name = name
        self.model = AzureOpenAIModels()
        self.system_prompt_template = system_prompt_template
        self.prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATES[prompt_template])
        self.instruction_prompt = self.prompt_template

    def run(self, input_data):
        """Executes the agent with the given input and returns structured output."""
        formatted_prompt = self.instruction_prompt.format(input=input_data) #get the user input into a chosen tamplate
        try:
            agent_output = self.model.get_chat_response(self.system_prompt_template, formatted_prompt)
        except:
            import base64
            formatted_prompt = base64.b64encode(formatted_prompt.encode()).decode()
            agent_output = self.model.get_chat_response(self.system_prompt_template, formatted_prompt)
        return agent_output
