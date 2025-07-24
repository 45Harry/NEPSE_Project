import streamlit as st
import sys
import os

# Ensure the parent directory is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import SentimentAnalysisUsingFinBert

st.title('Stock News Sentiment Analysis')
st.write('Enter a news headline or article in Nepali or English:')

user_input = st.text_area('Text to analyze', height=100)

if st.button('Analyze Sentiment'):
    if user_input.strip():
        with st.spinner('Analyzing...'):
            translated_text, sentiment = SentimentAnalysisUsingFinBert(user_input)
        
        st.subheader("Translated Text (English):")
        st.write(translated_text)

        st.success(f'Sentiment: {sentiment}')
    else:
        st.warning('Please enter some text.')
