from src.translator import Translator_nepali_to_english
from src.train_model import  Train_model
from src.predict import  load_model_and_predict
#from secret_key import HUGGINGFACE_API_KEY



def predict_sentimetn(text):
    """ 
    Input: Nepali news in string format 
    Output: Sentiment — "Positive", "Neutral", or "Negative" 
    Steps: 
        1. Translate to English 
        2. Preprocess text 
        3. Vectorize 
        4. Predict using trained model 
    """ 
    translated_text = Translator_nepali_to_english(text)
    sentiment = load_model_and_predict(translated_text) 



    return translated_text, sentiment


if __name__ == '__main__':
    translated, sentiment = predict_sentimetn('शेयर बजार आज तेजीमा छ।')
    print(f"Translated: {translated}")
    print(f"Sentiment: {sentiment}")