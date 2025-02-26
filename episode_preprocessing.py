import pandas as pd
import re

def remove_timestamps(text):
    # Regex pattern for timestamps (supports hh:mm:ss, mm:ss)
    pattern = r'\(\d{1,2}:\d{2}(?::\d{2})?\)'
    cleaned_text = re.sub(pattern, '', text).strip()
    return cleaned_text

import re

def clean_podcast_description(text):
    """Cleans a podcast description by removing timestamps, promotional content, social media handles, and unnecessary text."""

    # 1. Remove timestamps (e.g., "0:00", "11:45", "37:18")
    text = re.sub(r"\d{1,2}:\d{2}(\s?[APap][Mm])?", "", text)

    # 2. Remove promotional sections based on common keywords
    promo_keywords = ["SUBSCRIBE:", "FOLLOW US:", "OTHER SMOSHES:", "Tour Dates!", "Merch:", "By:", 
                      "Gametime:", "ZocDoc:", "Valor Recovery:", "Music:", "Find Theo:", "GET IN TOUCH:", 
                      "Referenced in the show", "Credit"]
    text = re.split("|".join(promo_keywords), text, maxsplit=1)[0]

    # 3. Remove social media handles (e.g., @username) and hashtags (e.g., #ExampleHashtag)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # 4. Remove contact info (emails and phone numbers)
    text = re.sub(r"\S+@\S+", "", text)  # Emails
    text = re.sub(r"\b\d{5,}\b", "", text)  # Phone numbers (assumes long numbers)

    # 5. Remove unnecessary references (e.g., "If you enjoy X, you’ll enjoy Y")
    text = re.sub(r"If you enjoy.*", "", text)

    # 6. Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


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

    url_pattern = re.compile(r'https?://\S+|www\.\S+|\(\s*https?://[^\s)]+\s*\)')  # Matches URLs in different formats
    non_english_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?\'\"-]')  # Matches non-English characters
    timestamps_pattern = r'\(\d{1,2}:\d{2}(?::\d{2})?\)'

    # Columns to clean (update with correct column names)
    columns_to_clean = ["description", "show.description"]  # Replace with actual column names

    # Apply regex cleaning
    for col in columns_to_clean:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(emoji_pattern, '', x))  # Remove emojis
        df[col] = df[col].apply(lambda x: re.sub(url_pattern, '', x))  # Remove URLs
        df[col] = df[col].apply(lambda x: re.sub(timestamps_pattern, '', x))
        df[col] = df[col].apply(clean_podcast_description)

    df = df[df["description"].apply(lambda x: len(re.findall(non_english_pattern, str(x))) < len(str(x)) * 0.3)]
    df = df[df["show.description"].apply(lambda x: len(re.findall(non_english_pattern, str(x))) < len(str(x)) * 0.3)]

    df = df.dropna()

    df['to_embed'] = "episode name: " + df['episodeName'] + ".\npodcast description: "  + df['description']
    df.to_csv('preprocessed_episodes.csv')

if __name__ == "__main__":
    preprocess_episodes()