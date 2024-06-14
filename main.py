import spacy
from textblob import TextBlob

# Load the small English language model in spaCy
nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using TextBlob.

    Parameters:
    text (str): The text to analyze.

    Returns:
    str: 'Positive', 'Negative', or 'Neutral' based on the sentiment polarity.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def extract_key_phrases(text):
    """
    Extract key phrases from the given text using spaCy.

    Parameters:
    text (str): The text to analyze.

    Returns:
    list: A list of key phrases (noun chunks) from the text.
    """
    doc = nlp(text)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    return key_phrases

def assess_quality(sentiment, key_phrases):
    """
    Assess the overall quality of a response based on sentiment and key phrases.

    Parameters:
    sentiment (str): The sentiment of the text ('Positive', 'Negative', 'Neutral').
    key_phrases (list): The list of key phrases extracted from the text.

    Returns:
    str: 'High', 'Medium', or 'Low' indicating the quality of the response.
    """
    if sentiment == "Positive" and len(key_phrases) > 3:
        return "High"
    elif sentiment == "Negative" or len(key_phrases) < 2:
        return "Low"
    else:
        return "Medium"

def main():
    """
    Main function to read responses from a file, analyze each response for sentiment and key phrases,
    and print the analysis results along with an overall quality assessment.
    """
    # Open and read the responses from the file
    with open('responses.txt', 'r') as file:
        responses = file.readlines()

    # Iterate through each response and analyze it
    for i, response in enumerate(responses):
        response = response.strip()  # Remove any leading/trailing whitespace
        sentiment = analyze_sentiment(response)
        key_phrases = extract_key_phrases(response)
        quality = assess_quality(sentiment, key_phrases)
        
        # Print the analysis results for each response
        print(f"Response {i+1}:")
        print(f"Text: {response}")
        print(f"Sentiment: {sentiment}")
        print(f"Key Phrases: {key_phrases}")
        print(f"Overall Quality: {quality}")
        print("")

if __name__ == "__main__":
    main()
