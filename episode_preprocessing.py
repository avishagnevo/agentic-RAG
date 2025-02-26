import pandas as pd
import re

def remove_timestamps(text):
    # Regex pattern for timestamps (supports hh:mm:ss, mm:ss)
    pattern = r'\(\d{1,2}:\d{2}(?::\d{2})?\)'
    cleaned_text = re.sub(pattern, '', text).strip()
    return cleaned_text

import re

import re

def clean_podcast_description(text):
    """Cleans a podcast description by removing timestamps, promotional content, social media handles, and unnecessary text."""

    # 1. Remove timestamps (e.g., "0:00", "11:45", "37:18")
    text = re.sub(r"\d{1,2}:\d{2}(\s?[APap][Mm])?", "", text)

    promo_keywords = [
        "SUBSCRIBE:", "FOLLOW US:", "OTHER SMOSHES:", "Tour Dates!", "Merch:", "By:", 
        "Gametime:", "ZocDoc:", "Valor Recovery:", "Music:", "Find Theo:", "GET IN TOUCH:", 
        "Referenced in the show", "Credit", "Go See", "Watch .* on YouTube", "Support .* @",
        "Get Merch @", "Download the .* app", "Visit .* today", "Use code .* for",
        "Follow me on social media", "Follow", "Tik Tok:", "Instagram:", "Text .* to \d{3}-\d{3}-\d{4}",  # Added for phone numbers like 'Text PODCAST'
        "PRE-ORDER .*", "THE VAULT", "BET-DAVID CONSULTING", "VALUETAINMENT UNIVERSITY",
        "NOW AVAILABLE!", "WATCH US AT .*", ".*\.COM", "FREE breakfast for life", "Learn more about your ad choices",
        "Subscribe to Save.*", "save \d+%", ".*% off", ".*subscription order", "Call \d{3}-\d{3}-\d{4}",  # Added for sales and phone numbers
        ".*on Apple Podcasts", ".*on Spotify", ".*on Rumble", "Subscribe to .*", "Join .* on Locals"
    ]
    text = re.split("|".join(promo_keywords), text, maxsplit=1, flags=re.IGNORECASE)[0]

    # 3. Remove social media handles (e.g., @username) and hashtags (e.g., #ExampleHashtag)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # 4. Remove emails and websites
    text = re.sub(r"\S+@\S+", "", text)  # Remove emails
    text = re.sub(r"https?://\S+", "", text)  # Remove URLs

    # 5. Remove call-to-action phrases (e.g., "send an email to", "Support this podcast")
    text = re.sub(r"send an email to.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Support this podcast.*", "", text, flags=re.IGNORECASE)

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

    df["episodeUri"] = "https://open.spotify.com/episode/" + df["episodeUri"]
    df["showUri"] = "https://open.spotify.com/show/" + df["showUri"]
    df['duration_ms'] = df['duration_ms'] / 60000
    df.set_index('id')

    # Create a new column with the word count of each description
    df['word_count'] = df['description'].str.split().str.len()

    # Filter out rows where the word count is less than 5
    df = df[df['word_count'] >= 5]

    # Optionally, drop the 'word_count' column if you no longer need it
    df = df.drop(columns=['word_count'])

    df.to_csv('preprocessed_episodes.csv')
    print(df.columns)

if __name__ == "__main__":
    preprocess_episodes()