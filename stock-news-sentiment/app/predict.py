import tensorflow as tf
from transformers import BertTokenizer
import os

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress most TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_model_and_predict(texts):
    try:
        # 1. Load tokenizer (from separate directory)
        #tokenizer = BertTokenizer.from_pretrained("../models/finbert_tf2_tokenizer")
        tokenizer = BertTokenizer.from_pretrained('/home/harry/Documents/Code/Data_Science/DeepLearning/MIT/NEPSE_Project/stock-news-sentiment/models/finbert_tf2_tokenizer')
        
        # 2. Load the TensorFlow SavedModel
        #model = tf.saved_model.load("../models/finbert_tf2")
        model = tf.saved_model.load('/home/harry/Documents/Code/Data_Science/DeepLearning/MIT/NEPSE_Project/stock-news-sentiment/models/finbert_tf2')
        
        # 3. Get the concrete function for predictions
        predict_fn = model.signatures["serving_default"]
        
        # 4. Tokenize input (with proper TensorFlow conversion)
        inputs = tokenizer(text, 
                         return_tensors="tf", 
                         padding=True, 
                         truncation=True,
                         max_length=512)
        
        # Convert to format expected by SavedModel
        input_dict = {
            "input_ids": tf.cast(inputs["input_ids"], tf.int32),
            "attention_mask": tf.cast(inputs["attention_mask"], tf.int32),
            "token_type_ids": tf.cast(inputs["token_type_ids"], tf.int32)
        }
        
        # 5. Make prediction
        outputs = predict_fn(**input_dict)
        logits = outputs['logits']
        
        # 6. Process output
        prediction = tf.argmax(logits, axis=-1).numpy()[0]
        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return sentiment_labels.get(prediction, "Unknown")
    
    except Exception as e:
        return f"Prediction error: {str(e)}"

if __name__ == '__main__':
    print(load_model_and_predict('The stock market is in crucified today.'))