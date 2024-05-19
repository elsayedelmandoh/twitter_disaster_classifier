import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import joblib
import requests
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Load Naive Bayes model
model_nb = joblib.load('model_nb.pkl')

# Load LSTM model
model_lstm = load_model("model_lstm.h5")

# Load Hugging Face model
API_TOKEN = "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"
model_url = "distilbert-base-uncased-finetuned-sst-2-english"

# Function to preprocess text
def preprocessing_data(texts):
    stop_words = stopwords.words('english')
    excluding= ['againts','no' ,'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
            "didn't",'doesn', "doesn't", 'hadn', "hadn't", 'has', "hasn't", 'haven', "haven't", 'isn', 
            "isn't", 'might', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shouldn', "shouldn't", 
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    stop_words= [word for word in stop_words if word not in excluding]
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    cleaned_texts = []  
    for sent in texts: 
        filtered_sent = []
        sent_no_links = re.sub(r'http[s]?://[^\s]+', '', sent) 
        tokens = word_tokenize(sent_no_links.lower())
        for token in tokens:
            token = re.sub(r'\W', '', token)
            if (not token.isnumeric()) and (len(token) > 2) and (token not in stop_words):
                filtered_sent.append(lemmatizer.lemmatize(stemmer.stem(token)))
        text = " ".join(filtered_sent)
        cleaned_texts.append(text)
    return cleaned_texts

# Function to query Hugging Face model
def query(prompt, model_url):
    API_URL = f"https://api-inference.huggingface.co/models/{model_url}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    label = response.json()[0][0]['label']
    score = response.json()[0][0]['score']
    return label, score

# Function to preprocess text for LSTM
def preprocess_for_lstm(texts):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    x_test_seq = tokenizer.texts_to_sequences(texts)
    x_test_pad = pad_sequences(x_test_seq, maxlen=100)
    return x_test_pad

# Streamlit app
def main():
    st.title("Disaster Tweet Classifier")

    # Sidebar for user input
    user_input = st.text_input("Enter a tweet:")
    if st.button("Classify"):
        # Preprocess input text
        cleaned_text = preprocessing_data([user_input])

        # Naive Bayes prediction
        nb_prediction = model_nb.predict(cleaned_text)
        st.write(f"Naive Bayes Prediction: {nb_prediction[0]}")

        # Hugging Face prediction
        hf_label, hf_score = query(cleaned_text, model_url)
        st.write(f"Hugging Face Prediction: {hf_label} (Score: {hf_score:.4f})")

        # LSTM prediction
        lstm_input = preprocess_for_lstm(cleaned_text)
        lstm_prediction = model_lstm.predict(lstm_input)
        st.write(f"LSTM Prediction: {np.round(lstm_prediction)[0]}")

if __name__ == "__main__":
    main()
