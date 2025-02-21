import pandas as pd
import re

def preprocess_episodes():
    df = pd.read_csv("top_podcasts.csv")
    df['id'] = 'e' + (df.index + 1).astype(str)

    # Filter regions
    df = df[(df['region'] == 'us') | (df['region'] == 'gb') | (df['region'] == 'au')]
    df = df[(df['language'] == 'en') | (df['language'] == 'en-US') | (df['language'] == 'en-GB') 
            | (df['language'] == 'en-AU')]

    df = df[['id', 'episodeUri', 'showUri', 'episodeName', 'description', 'show.name',
            'show.description', 'show.publisher', 'duration_ms']]

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        u"\U0001F700-\U0001F77F"  # Alchemical Symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed Characters
        "]+", flags=re.UNICODE)

    url_pattern = re.compile(r"https?://\S+|www\.\S+")  # Matches URLs
    non_english_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?\'\"-]')  # Matches non-English characters

    # Columns to clean (update with correct column names)
    columns_to_clean = ["description", "show.description"]  # Replace with actual column names

    # Apply regex cleaning
    for col in columns_to_clean:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(emoji_pattern, '', x))  # Remove emojis
        df[col] = df[col].apply(lambda x: re.sub(url_pattern, '', x))  # Remove URLs

    df = df[df["description"].apply(lambda x: len(re.findall(non_english_pattern, str(x))) < len(str(x)) * 0.3)]
    df = df[df["show.description"].apply(lambda x: len(re.findall(non_english_pattern, str(x))) < len(str(x)) * 0.3)]

    df = df.dropna()

    df['to_embed'] = "episode name: " + df['episodeName'] + ".\npodcast description: "  + df['description']
    df.to_csv('preprocessed_episodes.csv')

if __name__ == "__main__":
    preprocess_episodes()