import json
import re
import ast


def check_query_pass(query_pass):
    """
    Check the query pass and return a dictionary
    :param query_pass: query pass
    """
    try:
        query_pass_dict = json.loads(re.sub(r"```json\n?|```", "", query_pass).strip())
        keys_checker = [query_pass_dict["pass"], query_pass_dict["reason"]]
    except:
        query_pass_dict = {"pass": True, "reason": "agent failed to parse query"}

    return query_pass_dict

def check_search_filters(search_filters):
    """
    Check the search filters and return a dictionary
    :param search_filters: search filters
    """
    try:
        search_filters = json.loads(re.sub(r"```json\n?|```", "", search_filters).strip())
        keys_checker = [search_filters["dataset"], search_filters["recommendation_amount"],
                        search_filters["duration_range"]]
    except:
        print("Not json format")
        try: # try dict format
            # Insert quotes around keys: match word characters before a colon.
            fixed_str = re.sub(r"(\w+)\s*:", r'"\1":', search_filters)
            # Insert quotes around unquoted string values if needed. This part can be trickier;
            # for simplicity, assume that values like 'episodes' and 'not_limited' need quotes.
            fixed_str = re.sub(r':\s*([a-zA-Z_]+)([,\}])', r': "\1"\2', fixed_str)
            search_filters = ast.literal_eval(fixed_str)
            keys_checker = [search_filters["dataset"],
                            search_filters["recommendation_amount"],
                            search_filters["duration_range"]]
        except:
            search_filters = {"dataset": "both", "recommendation_amount": 3,
                          "duration_range": "not_limited"}
            print("Agent failed to parse search filters, using default values.")

    if search_filters["duration_range"] == "short":
        search_filters["range"] = [0, 15]
    elif search_filters["duration_range"] == "mid":
        search_filters["range"] = [15, 45]
    elif search_filters["duration_range"] == "mid high":
        search_filters["range"] = [45, 90]
    elif search_filters["duration_range"] == "long":
        search_filters["range"] = [90, 1000]
    else:
        search_filters["range"] = [0, 1000]

    if search_filters["recommendation_amount"] > 10:
        print(f"Our agent approximated your recommendation amount as {search_filters['recommendation_amount']},"
              " but we can only provide up to 10 recommendations.")
        search_filters["recommendation_amount"] = 10

    if search_filters["dataset"] == "podcast":
        search_filters["Pinecone Format"] = {"dataset": {"$eq": "podcasts"}}
    elif search_filters["dataset"] == "episodes":
        lower, upper = search_filters["range"]
        search_filters["Pinecone Format"] = {"dataset": {"$eq": "episodes"},
                                             "duration_min": {"$gte": lower, "$lte": upper}}
    elif search_filters["dataset"] == "both":
        lower, upper = search_filters["range"]
        search_filters["Pinecone Format"] = {"$or": [
            {"dataset": {"$eq": "podcasts"}},
            {"dataset": {"$eq": "episodes"}, "duration_min": {"$gte": lower, "$lte": upper}}
        ]}
    else:
        search_filters["Pinecone Format"] = {}

    return search_filters