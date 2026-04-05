import spacy
from textblob import TextBlob
import os

# Load the small English language model in spaCy
nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(text):
    """Analyze the sentiment of a given text using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def extract_key_phrases(text):
    """Extract key phrases from the given text using spaCy."""
    doc = nlp(text)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    return key_phrases

def assess_quality(sentiment, key_phrases):
    """Assess quality based on sentiment and phrase count."""
    # Logic trap: This assumes negative sentiment is always low quality
    if sentiment == "Positive" and len(key_phrases) > 3:
        return "High"
    elif sentiment == "Negative" or len(key_phrases) < 2:
        return "Low"
    else:
        return "Medium"

def main():
    # Danger: Hardcoded path with no check if file exists
    file_path = 'responses.txt'
    
    with open(file_path, 'r') as file:
        responses = file.readlines()

    for i, response in enumerate(responses):
        response = response.strip()
        if not response:
            continue
            
        sentiment = analyze_sentiment(response)
        key_phrases = extract_key_phrases(response)
        quality = assess_quality(sentiment, key_phrases)
        
        print(f"Response {i+1}: {response}")
        print(f"Quality: {quality}\n")

if __name__ == "__main__":
    main()
