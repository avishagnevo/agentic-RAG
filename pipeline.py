import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import utils
from agent import Agent
import database


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

        # Database Selection
        query_pass = self.agents["QueryInitialCheck"].run(user_query)
        query_pass_dict = utils.check_query_pass(query_pass)

        if query_pass_dict["pass"] == False:
            return str("Our agent think that your prompt needs some modification for"
                       " helping us to chose your desired podcast.\nAgent reason:\n" + query_pass_dict["reason"])


        # Index Filters Extraction
        search_filters = self.agents["SearchFilters"].run(user_query)
        search_filters = utils.check_search_filters(search_filters)

        # Summarization of User Needs
        needs_summary = self.agents["NeedUnderstanding"].run(user_query)

        # Semantic Search using Pinecone, get 3 times the recommendation amount to account for filtering
        # search_filters["Pinecone Format"] is a dictionary that contains the filters in the format required by Pinecone
        query_embedding = self.embedding_model.get_query_embedding(needs_summary)
        search_results = self.index.retrieve_data(query_embedding, top_k=3 * search_filters["recommendation_amount"],
                                                  filters=search_filters["Pinecone Format"])

        # Augmented Prompt Construction
        augmented_prompt = {
            "search_results": [
                {"id": podcast["id"], "text": podcast["metadata"].get("text", "")}
                for podcast in search_results
            ] if 'dataset' in search_filters["Pinecone Format"].keys() and search_filters["Pinecone Format"]['dataset']['$eq'] != 'episodes' else
            [
                {"id": podcast["id"], "text": podcast["metadata"].get("text", ""), "duration": podcast["metadata"].get("duration_min", "")}
                for podcast in search_results
            ],
            "recommendation_amount": search_filters["recommendation_amount"],
            "user_query": user_query,
            "needs_summary": needs_summary
        }

        podcast_selection = self.agents["Selector"].run(augmented_prompt)
        selected_ids = utils.check_selector_output(podcast_selection, search_results, search_filters)


        selected_data = []
        # Construct the final output data based on the selected podcast IDs
        for result in search_results:
            if result["id"] in selected_ids:
                entry = { }
                if result["metadata"].get("dataset", "") == "episodes":
                    entry["Title"] = result["metadata"].get("episode_name", "")
                    entry["description"] = result["metadata"].get("episode_description", "")
                    entry["URL"] = result["metadata"].get("episode_url", "")
                    entry["duration_min"] = result["metadata"].get("duration_min", None)
                    entry["Type"] = "Episode"
                else:
                    entry["Title"] = result["metadata"].get("title", "")
                    entry["description"] = result["metadata"].get("description", "")
                    entry["URL"] = result["metadata"].get("itunes_url", "")
                    entry["Type"] = "Podcast"
                selected_data.append(entry)

        # Response Generation
        response = self.agents["ResponseGeneration"].run(selected_data)

        # Supervision & Refinement
        final_response = self.agents["Supervision"].run("original_prompt: " + str(user_query) +
                                                        "\n" + "final_response: " + str(response))

        return final_response

def initialize_index():
    """
    Initialize the Pinecone index for the podcast dataset.
    """
    index, dataset, embedding_model = database.init_database()
    return index, dataset, embedding_model

def run_pipeline(index, dataset, embedding_model, user_prompt):
    pipeline = AgenticPipeline(index, dataset, embedding_model)
    final_output = pipeline.execute(user_prompt)
    print(f"\n##########################################################################\n")
    print(final_output)
    print(f"\n##########################################################################")


if __name__ == "__main__":
    print(
        "Hello! I’m your AI-powered podcast recommendation agent.\n"
        "I specialize in finding the best podcasts tailored to your interests. "
        "Just tell me what you're looking for—whether it’s about tech, business, health, or any topic—and I'll curate the perfect list for you. "
        "I also consider factors like your available time and specific preferences to provide the best experience.\n"
        "Go ahead, enter a prompt like: 'Find me top Data Science podcasts.'\n"
    )
    user_input = input("Type here: ")
    # user_input = "Find me top Data Science podcasts."
    # user_input = "I want one podcast about photography"
    # user_input = "I want knowledge on karate"
    # user_input = "I love eating pizza"
    # user_input = "I want to hear about the history of israel"
    # user_input = "I want podcast that will teach me how to rob a bank"
    # user_input = "I've just adopted a new puppy and I want to learn how to train it, and all the important things I need to know about raising a puppy"
    # user_input = "I want one episode about Elon Musk"
    index, dataset, embedding_model = initialize_index()
    run_pipeline(index, dataset, embedding_model, user_input)