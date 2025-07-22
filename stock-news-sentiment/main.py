from src.translator import Translator_nepali_to_english
from src.train_model import  Train_model
from src.predict import  load_model_and_predict


def SentimentAnalysisUsingFinBert(text):
    text = Translator_nepali_to_english(text)
    text = load_model_and_predict(text) 



    return text


if __name__ == '__main__':
    sentiment = SentimentAnalysisUsingFinBert('शेयर बजार आज तेजीमा छ।')