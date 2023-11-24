# Streamlit App Structure: Import the necessary libraries.

import streamlit as st
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# User Input: Use Streamlit to get user input

st.title('NLP Processor and Sentiment Analysis')
text = st.text_area("Enter Text Here:")

# Tokenization: Tokenize the user's input text

def tokenize(text):
    blob = TextBlob(text)
    return blob.words

if st.button('Tokenize'):
    st.subheader('Tokenized Text')
    tokens = tokenize(text)
    st.write(tokens)
    
    
# Stop Word Removal: Remove stop words from the tokenized text    
    
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

if st.button('Remove Stop Words'):
    st.subheader('Text without Stop Words')
    filtered_tokens = remove_stopwords(tokenize(text))
    st.write(filtered_tokens)

# Lemmatization: Lemmatize the text to get the base or dictionary form of words
    
def lemmatize(tokens):
    blob = TextBlob(' '.join(tokens))
    return [word.lemmatize() for word in blob.words]

if st.button('Lemmatize'):
    st.subheader('Lemmatized Text')
    lemmatized_tokens = lemmatize(remove_stopwords(tokenize(text)))
    st.write(lemmatized_tokens)

    
# Sentiment Analysis: Use TextBlob to perform sentiment analysis

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment

if st.button('Analyze Sentiment'):
    st.subheader('Sentiment Analysis')
    sentiment = analyze_sentiment(text)
    st.write(f"Sentiment Polarity: {sentiment.polarity}")
    st.write(f"Sentiment Subjectivity: {sentiment.subjectivity}")

    
# Display Findings and Insights: After each NLP process, show findings or insights

# Frequency Distribution: Show the most common words in the text

from collections import Counter

def show_frequency_distribution(tokens):
    st.subheader('Frequency Distribution')
    frequency_dist = Counter(tokens)
    most_common_words = frequency_dist.most_common(10)
    st.write("Most common words:", most_common_words)
    st.bar_chart(dict(most_common_words))

if st.button('Show Frequency Distribution'):
    filtered_tokens = remove_stopwords(tokenize(text))
    show_frequency_distribution(filtered_tokens)
    
# Part-of-Speech Tagging: Display the part-of-speech tags of the words in the text.

def show_pos_tags(text):
    blob = TextBlob(text)
    st.subheader('Part-of-Speech Tags')
    pos_tags = blob.tags
    st.write(pos_tags)

if st.button('Show POS Tags'):
    show_pos_tags(text)
    
# N-Grams: Find and display N-grams in the text

def show_n_grams(text, n=2):
    blob = TextBlob(text)
    st.subheader(f'{n}-Grams')
    n_grams = blob.ngrams(n=n)
    st.write(n_grams)

n_gram_number = st.slider("Choose N for N-grams", min_value=2, max_value=5, value=2)
if st.button('Show N-Grams'):
    show_n_grams(text, n_gram_number)

    
# Word Length Distribution: Visualize the distribution of word lengths

def show_word_length_distribution(tokens):
    st.subheader('Word Length Distribution')
    word_lengths = [len(word) for word in tokens]
    word_length_count = Counter(word_lengths)
    st.bar_chart(word_length_count)

if st.button('Show Word Length Distribution'):
    tokens = tokenize(text)
    show_word_length_distribution(tokens)

# Sentiment Over Time: If the text is divided into sentences or sections, show how sentiment changes

def show_sentiment_over_time(text):
    blob = TextBlob(text)
    sentiments = [sentence.sentiment.polarity for sentence in blob.sentences]
    st.subheader('Sentiment Over Time')
    st.line_chart(sentiments)

if st.button('Show Sentiment Over Time'):
    show_sentiment_over_time(text)
    
# Unique Words: Count and display the number of unique words
def show_unique_words(tokens):
    st.subheader('Unique Words')
    unique_words = set(tokens)
    st.write(f"Total unique words: {len(unique_words)}")

if st.button('Show Unique Words'):
    tokens = tokenize(text)
    show_unique_words(tokens)

