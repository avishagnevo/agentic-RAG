import utils
from agent import Agent
import database
import json
import re


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
            "QueryInitialCheck": Agent("QueryInitialCheck", "structured", "FewShot"),
            "SearchFilters": Agent("SearchFilters", "structured", "FewShot"),
            "NeedUnderstanding": Agent("NeedUnderstanding", "structured", "InstructionBased"),
            "Selector": Agent("Selector", "structured", "InstructionBased"),
            "ResponseGeneration": Agent("ResponseGeneration", "structured", "InstructionBased"),
            "Supervision": Agent("Supervision", "structured", "PersonaPattern"),
        }
    
    def execute(self, user_query):
        """
        Runs the full agentic pipeline, ensuring each step feeds into the next logically.
        """
        print("Starting Agentic Pipeline Execution...")

        # Database Selection
        query_pass = self.agents["QueryInitialCheck"].run(user_query) #should return an output like a Pinecone filter
        print(query_pass)
        query_pass_dict = utils.check_query_pass(query_pass)

        if query_pass_dict["pass"] == False:
            return str("Our agent think that your prompt needs some modification for"
                       " helping us to chose your desired podcast.\nAgent reason:\n" + query_pass_dict["reason"])


        # User Input Processing
        processed_input = user_query # TODO Implement a function, should be handled without an agant
        # print("Processed Input:", processed_input)
        
        # Index Filters Extraction
        search_filters = self.agents["SearchFilters"].run(processed_input) #should return an output like a Pinecone filter
        print(search_filters)
        search_filters = utils.check_search_filters(search_filters)

        print("Search Filters:", search_filters)


        # Need Understanding
        needs_summary = self.agents["NeedUnderstanding"].run(processed_input)
        print("Needs Summary:", needs_summary)


        # Semantic Search using Pinecone
        query_embedding = self.embedding_model.get_query_embedding(needs_summary)
        search_results = self.index.retrieve_data(query_embedding, top_k=3 * search_filters["recommendation_amount"],
                                                  filters=search_filters["Pinecone Format"])

        # Augmented Prompt Construction
        augmented_prompt = {
            "search_results": [
                {"id": podcast["id"], "text": podcast["metadata"].get("text", "")}
                for podcast in search_results
            ],
            "recommendation_amount": search_filters["recommendation_amount"],
            "processed_input": processed_input,
            "needs_summary": needs_summary
        }

        print("Augmented Prompt:", augmented_prompt)

        podcast_selection = self.agents["Selector"].run(augmented_prompt)

        selected_ids = utils.check_selector_output(podcast_selection, search_results, search_filters)


        selected_data = []
        for result in search_results:
            if result["id"] in selected_ids:
                title = result["metadata"].get("title", result["metadata"].get("episodeName", ""))
                entry = {
                    "id": result["id"],
                    "title": title,
                    "text": result["metadata"].get("text", "")
                }
                # For episodes, include duration_min; for podcasts, include itunes_url if available.
                if result["metadata"].get("dataset", "") == "episodes":
                    entry["duration_min"] = result["metadata"].get("duration_min", None)
                else:
                    entry["itunes_url"] = result["metadata"].get("itunes_url", "")
                selected_data.append(entry)

        # Convert the selected data into a JSON string for the agent
        response_input = json.dumps(selected_data)

        # Run the ResponseGeneration agent with the constructed input
        response = self.agents["ResponseGeneration"].run(response_input)

        print("Generated Response:", response)


        # Supervision & Refinement
        final_response = self.agents["Supervision"].run("original_prompt: "+ str(user_query) +
                                                        "\n" + "final_response: " + str(response))
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
    print(f"\n\n\n\n\n##########################################################################")
    print(final_output)
         

if __name__ == "__main__":
    # user_prompt = "Find me top Data Science podcasts."
    # user_prompt = "I love eating pizza"
    # user_prompt = "I want knowledge on karate"
    # user_prompt = "Help me to learn coding in my coffee break tomorrow"
    user_prompt = "I want one podcast about photography"

    index, dataset, embedding_model = initialize_index()
    run_pipeline(index, dataset, embedding_model, user_prompt)