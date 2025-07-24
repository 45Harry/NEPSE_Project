import pickle
from src.translator import Translator_nepali_to_english
from src.train_model import Train_model


def load_model_and_predict(text): #(pickle_file, text):
    # Load the model from the pickle file
    with open('./models/sentiment_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Preprocess the input text if necessary
    # (This step may vary depending on your model's requirements)
    # For example, you might need to clean the text, tokenize it, etc.

    #text = Translator_nepali_to_english(text)
    train_model = Train_model()
    text = train_model.preprocess_financial_text(text)


    
    # Make a prediction
    prediction = model.predict([text])  # Assuming the model expects a list of texts
    
    return prediction