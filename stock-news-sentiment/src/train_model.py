# Loading the Nesessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import spacy
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import subprocess
import sys
from tqdm import tqdm



# Process with spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)



import os
from hf_token_key import HUGGINGFACE_API_KEY

token = HUGGINGFACE_API_KEY


class Train_model:

    def load_data(self,filepath):
        """
         Load the dataset from the specified file path.
         Returns: pandas.DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            print('Dataset loaded successfully :)')
            #print(f"Data shape: {df.shape}")
            #print(f"Columns: {df.columns.tolist()}")

            return df

        except Exception as e :
            print(f"Unable to load the Dataset {e}")
            raise


    def fix_contractions(self,text):
        # Custom contraction mapping
        CONTRACTION_MAP = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
            "it's": "it is",
            "that's": "that is"
        }
        for contraction, expansion in CONTRACTION_MAP.items():
            text = text.replace(contraction, expansion)
        return text





    def preprocess_financial_text(self, text, keep_negs=True, keep_numbers=True, keep_currency=True):
        """
        Preprocess financial text with special handling for:
        - Currency symbols and amounts ($, Rs, NPR)
        - Nepali news content (basic handling)
        - Financial terminology preservation
        """

        # Expand contractions
        text = self.fix_contractions(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Special handling for financial amounts before tokenization
        if keep_currency:
            # Normalize currency representations
            text = re.sub(r'\$(\d+\.?\d*)', r'dollar_\1', text)  # $100 → dollar_100
            text = re.sub(r'(\d+\.?\d*)\s?dollars?', r'dollar_\1', text, flags=re.IGNORECASE)
            text = re.sub(r'Rs\.?\s?(\d+\.?\d*)', r'rupee_\1', text)  # Rs 100 → rupee_100
            text = re.sub(r'NPR\s?(\d+\.?\d*)', r'npr_\1', text)  # NPR 100 → npr_100

        # Process with spaCy

        doc = nlp(text)

        processed_tokens = []
        for token in doc:
            if token.is_space:
                continue

            # Keep currency terms as single tokens
            if keep_currency and token.text.lower() in ['$', 'rs', 'npr', 'dollar', 'rupee']:
                processed_tokens.append(token.text.lower())
                continue

            # Handle punctuation - keep only certain financial-relevant punctuation
            if token.is_punct and token.text not in ['%', '.']:  # Keep percentage and decimal points
                continue

            # Handle numbers based on parameter
            if not keep_numbers and token.like_num:
                continue

            # Handle stop words with negation preservation
            if token.is_stop and not (keep_negs and token.lower_ in ["not", "no", "never", "nor"]):
                continue

            # Lemmatization
            lemma = token.lemma_.lower().strip()

            # Special handling for financial terms
            if lemma in ['stock', 'share', 'price', 'market']:  # Don't lemmatize these
                lemma = token.text.lower()

            if lemma:
                processed_tokens.append(lemma)
        return " ".join(processed_tokens)




    def clean_data(self,df):
        """
        Clean and preprocess the DataFrame.
        Returns: pandas.DataFrame

        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")

            # Make a copy to avoid settingwihtcopywarinng
            clean_df = df.copy()

            clean_df = clean_df.drop_duplicates().dropna()
            # Convert sentiment labels to numerical values
            sentiment_map = {'neutral': 0, 'positive': 1, 'negative': 2}
            clean_df['label'] = clean_df['Sentiment'].map(sentiment_map)

            tqdm.pandas()
            clean_df['cleaned_sentences'] = df['Sentence'].apply(self.preprocess_financial_text)
            print("Data Cleaned Sucessfully")
            return clean_df

        except Exception as e:
            print(f"Error while Cleaning and Preprocessing the Dataset: {e}")
            raise




    def model_traning(self,df):
        """
        FineTuning the FINBert model for our financial datasets
        Tesnsorflow is required

        """

        # Check if GPU is available and configure TensorFlow
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Use the first available GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Invalid device or cannot modify logical devices once initialized.
                print(e)
        else:
            print("No GPU available, using CPU.")


        token = HUGGINGFACE_API_KEY ##

        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert",token=token)
        model = TFBertForSequenceClassification.from_pretrained("ProsusAI/finbert",token=token)

        # Load your CSV
        if 'cleaned_sentences' not in df.columns or 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'cleaned_sentences' and 'label' columns")

        texts = df['cleaned_sentences'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        #texts, labels = df['Sentence'].tolist(), df['label'].tolist() # changed Sentence to cleaned_sentences
        #texts, labels = df['cleaned_sentences'].tolist(), df['label'].tolist() # changed Sentence to cleaned_sentences

        try:
            def encode(text, label):
                text = tf.compat.as_str(text.numpy())  # bytes → str wrapper
                tokens = tokenizer(
                    text, max_length=128, truncation=True, padding="max_length"
                )
                return (
                    tf.convert_to_tensor(tokens["input_ids"], dtype=tf.int32),
                    tf.convert_to_tensor(tokens["attention_mask"], dtype=tf.int32),
                    label,
                )


            def tf_encode(text, label):
                input_ids, attention_mask, lbl = tf.py_function(
                    func=encode,
                    inp=[text, label],
                    Tout=[tf.int32, tf.int32, tf.int32]
                )
                label = tf.cast(lbl, tf.int64)
                input_ids.set_shape([None])
                attention_mask.set_shape([None])
                label.set_shape([])
                return {"input_ids": input_ids, "attention_mask": attention_mask}, label



            # Create tf.data.Dataset
            ds = tf.data.Dataset.from_tensor_slices((texts, labels))
            ds = ds.map(tf_encode).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

            # Split into train/val or use tfds
            train_data = int(0.8 * len(ds))
            ds_train = ds.take(train_data)  # example split
            ds_valid = ds.skip(train_data)

            model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
            )
            model.fit(ds_train, validation_data=ds_valid, epochs=3)
            print('Model finetuned sucessfully :)')

            return model,tokenizer

        except Exception as e:

            return(f'Error finetuning the model{e}')


    def save_model(self,model,tokenizer):
        """
        Saving the model as pickle file in models folders
        """
        try:
            # Save
            os.makedirs("finbert_tf", exist_ok=True)
            os.makedirs("finbert_tf_tokenizer", exist_ok=True)

            # SavedModel directory (Keras native)
            model.save("finbert_tf")

            # Tokenizer files/home/harry/Documents/MIT/NEPSE/stock-news-sentiment/data/english_dataset.csv
            tokenizer.save_pretrained("finbert_tf_tokenizer")

            print("\n✅  Saved model to   : ./finbert_tf2/")
            print("✅  Saved tokenizer to: ./finbert_tf2_tokenizer/")

        except Exception as e:
            return(f"Error saving the model:{e}")



if __name__ == '__main__':

    train_m = Train_model()

    # 1. Load data
    data_path = f'../data/english_dataset.csv'
    try:
        dataset = train_m.load_data(data_path)

        # 2. Verify loaded data
        #print("\nRaw data sample:")
        #print(dataset.head(2))

        # 3. Clean data
        print('Dataset Cleaning is starting....')
        cleaned_data = train_m.clean_data(dataset)
        print('Dataset Cleaning is Completed :)')

        # 4. Verify cleaned data
        #print("\nCleaned data sample:")
        #print(cleaned_data[['Sentence', 'cleaned_sentences', 'Sentiment', 'label']].head(2))

        # 5. Train model (uncomment when ready)
        print("Model Traning Started...")
        model, tokenizer = train_m.model_traning(cleaned_data)
        print("MOdel Fintuned Sucessfully :)")
        #train_m.save_model(model, tokenizer)

    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        print("Debugging tips:")
        print("- Verify the CSV file exists at the specified path")
        print("- Check the CSV contains 'Sentence' and 'Sentiment' columns")
        print("- Ensure pandas is properly installed (pip install pandas)")




