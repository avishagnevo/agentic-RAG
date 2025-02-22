from agent import Agent
import database
from llms import AzureOpenAIModels


class AgenticPipeline:
    """
    Manages the structured execution of AI agents to process user queries and retrieve relevant podcast recommendations.
    """
    def __init__(self, index, dataset, embedding_model):
        self.index = index
        self.dataset = dataset
        self.embedding_model = embedding_model
        self._initialize_agents()

    def _initialize_agents(self):
        """Initializes all agents using the pre-defined templates from agent_templates.json."""

        self.agents = {
            "QueryInitialCheck": Agent("QueryInitialCheck", "structured", "QuestionRefinementPattern"), # or "InstructionBased"
            "SearchFilters": Agent("SearchFilters", "structured", "FewShot"),
            "NeedUnderstanding": Agent("NeedUnderstanding", "structured", "PersonaPattern"),
            "ResponseGeneration": Agent("ResponseGeneration", "unstructured", "AudiencePattern"),
            "Supervision": Agent("Supervision", "structured", "FewShot"),
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
        processed_input = user_query # TODO Implement a function, should be handled without an agant
        print("Processed Input:", processed_input)
        
        # Step 2: Index Filters Extraction
        search_filters = self.agents["SearchFilters"].run(processed_input) #should return an output like a Pinecone filter
        print("Search Details:", search_filters)
        
        # Step 3: Need Understanding & Augmentation
        needs_summary = self.agents["NeedUnderstanding"].run(processed_input)
        print("Needs Summary:", needs_summary)
        
        # Step 4: Semantic Search using Pinecone
        query_embedding = self.embedding_model.get_query_embedding(needs_summary)
        search_results = self.index.retrieve_data(query_embedding, top_k=5, filters=search_filters)
        print("Semantic Search Results:", search_results)

        #maybe add another supervision here?
        
        # Step 5: Augmented Prompt Construction
        augmented_prompt = {
            "search_results": search_results,
            "needs_summary": needs_summary,
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
    index, dataset, embedding_model = database.init_database()
    return index, dataset, embedding_model

def run_pipeline(index, dataset, embedding_model, user_prompt):
    # Initialize and execute the pipeline
    pipeline = AgenticPipeline(index, dataset, embedding_model)
    final_output = pipeline.execute(user_prompt)
    print("Final Output:", final_output)
         

if __name__ == "__main__":
    # user_prompt = "Find me top Data Science podcasts."
    # user_prompt = "I really want to here about RAG"
    user_prompt = "Give me 54 optional podcasts episodes that talked about llms"
    # index, dataset, embedding_model = initialize_index()
    # run_pipeline(index, dataset, embedding_model, user_prompt)

    # llm test
    agent2 = Agent("SearchFilters", "structured", "FewShot")
    filtered_prompt = agent2.run(user_prompt)
    print(filtered_prompt)
    embedding_model = AzureOpenAIModels().embedding_model
    query_embedding = embedding_model.embed_query(filtered_prompt)
    print(query_embedding[:10]) # Show the first 10 characters of the first vector