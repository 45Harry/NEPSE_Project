from googletrans import Translator

def Translator_nepali_to_english(text):
    """
        Translates Nepali text into English using GoogleTranslate
    """


    if not text or not isinstance(text,str):
        raise ValueError("Input must be a non-empty string.")
    

    translator = Translator()
    translated = translator.translate(text, src='ne', dest='en')
    return(translated.text)  # Output: "Hola, ¿cómo estás?"



if __name__== '__main__':
    data = Translator("Hello my name is hari")
    print('this is the translated text ',data)





