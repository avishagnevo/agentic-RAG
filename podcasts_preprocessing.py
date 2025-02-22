import pandas as pd
import re
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0


def remove_emojis_and_symbols(text):
    emoji_symbol_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F700-\U0001F77F"  # Alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002500-\U00002BEF"  # Box drawing, Misc symbols
        u"\U00002700-\U000027BF"  # Dingbats (✔, ✖, etc.)
        u"\U000024C2-\U0001F251"  # Enclosed characters
        u"\U0000200D"  # Zero width joiner
        u"\U0001F004-\U0001F0CF"  # Playing cards
        u"\U0001F018-\U0001F270"  # Some extra pictographs
        u"\U0001F650-\U0001F67F"  # Ornamental dingbats
        "]+", flags=re.UNICODE)

    return emoji_symbol_pattern.sub('', text)

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|\(\s*https?://[^\s)]+\s*\)')  # Matches URLs in different formats
    return url_pattern.sub('', text).strip()

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False  # Handle cases where detection fails


if __name__ == "__main__":
    categories = pd.read_json('podcasts_data/categories.json', lines=True)
    podcasts = pd.read_json('podcasts_data/podcasts.json', lines=True)

    podcasts = podcasts[['podcast_id', 'itunes_url', 'title', 'description', 'average_rating']].dropna()
    categories = categories[['podcast_id', 'category']].dropna()
    categories_groped = categories.groupby('podcast_id').agg({'category': lambda x: ', '.join(x)}).reset_index()
    # Merge categories for each podcast
    podcasts_merged = pd.merge(podcasts, categories_groped, on='podcast_id')

    podcasts_merged["title"] = podcasts_merged["title"].astype(str).apply(remove_emojis_and_symbols)
    podcasts_merged["description"] = podcasts_merged["description"].astype(str).apply(remove_emojis_and_symbols)
    podcasts_merged["description"] = podcasts_merged["description"].apply(remove_html)
    podcasts_merged["title"] = podcasts_merged["title"].apply(remove_html)
    podcasts_merged["title"] = podcasts_merged["title"].astype(str).apply(remove_urls)
    podcasts_merged["description"] = podcasts_merged["description"].astype(str).apply(remove_urls)
    podcasts_merged["title"] = podcasts_merged["title"].str.replace(r'\s+', ' ', regex=True).str.strip()
    podcasts_merged["description"] = podcasts_merged["description"].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Leave podcast with more than 9 words in description
    podcasts_merged = podcasts_merged[podcasts_merged["description"].apply(lambda x: len(x.split()) > 9)]
    # Leave podcast with more than 4.5 in rating
    podcasts_merged = podcasts_merged[podcasts_merged["average_rating"] > 4.5]

    # Apply language filtering
    podcasts_merged["is_english"] = podcasts_merged["description"].astype(str).apply(is_english)
    podcasts_filtered = podcasts_merged[podcasts_merged["is_english"]].drop(columns=["is_english"])
    podcasts_filtered = podcasts_filtered.reset_index(drop=True)

    # Add id column to podcasts_filtered
    podcasts_filtered['id'] = 'p' + (podcasts_filtered.index + 1).astype(str)
    podcasts_filtered = podcasts_filtered.drop(columns=['podcast_id'])
    # Add to_embed column to podcasts_filtered
    podcasts_filtered['to_embed'] = "podcast title: " + podcasts_filtered['title'] + ". podcast description: " + \
                                    podcasts_filtered['description']
    # add column called 'dataset' to all rows with value 'podcasts'
    podcasts_filtered['dataset'] = 'podcasts'

    # save podcasts_filtered to podcasts_filtered.csv
    podcasts_filtered.to_csv('podcasts_filtered.csv', index=False)