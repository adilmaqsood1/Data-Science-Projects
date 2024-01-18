import streamlit as st
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer from the pickle file
with open('IMDb-Sentiment-Analysis-App/review_model.pkl', 'rb') as file:
    model, tokenizer = pickle.load(file)

# Streamlit app
st.title('IMDb Sentiment Analysis App')

# Input text area for user to enter a movie review
review_text = st.text_area('Enter your movie review:', '')

# Preprocess the user input using the tokenizer
sequence = tokenizer.texts_to_sequences([review_text])
padded_sequence = pad_sequences(sequence, maxlen=100)

# Make prediction using the loaded model
prediction = model.predict(padded_sequence)[0, 0]

# Display the sentiment prediction
sentiment = 'Positive' if prediction > 0.5 else 'Negative'
st.write(f'Sentiment Prediction: {sentiment} ({prediction:.4f})')

# Optionally, you can display the probability score as a progress bar
st.progress(prediction)

# Additional features or information can be added as needed
