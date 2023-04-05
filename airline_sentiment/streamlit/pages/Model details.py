
import streamlit as st
import pandas as pd
import pickle
# from airline_sentiment.ml_logic.data import clean
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unidecode
import contractions
import re

def clean(text, airport_code_list):

    for air_code in airport_code_list: # remove airline codes
        text = text.replace(air_code, ' ')

    text = text.lower() # Lower Case

    text = re.sub("@[A-Za-z0-9_]+","", text) # remove mentions
    text = re.sub("#[A-Za-z0-9_]+","", text) # remove hashtags

    text = re.sub(r"http\S+", "", text) # remove links
    text = re.sub(r"www.\S+", "", text) # remove links

    expanded_words = [contractions.fix(word) for word in text.split()] # remove contractions

    text = ' '.join(expanded_words) # join words

    unaccented_string = unidecode.unidecode(text) # remove accents

    tokenized = word_tokenize(unaccented_string) # Tokenize

    stop_words = set(stopwords.words('english')) # Make stopword list

    stop_word_to_keep = ['was', 'are', 'did', 'been', 'have', 'until', 'while', 'about', 'against', 'between', 'during', 'before', 'after', 'again', 'when', 'where', 'why', 'how', 'any', 'not', 'no','very', "aren't", "wasn't", "shouldn't", "should", "won't", "wouldn't"]

    stop_words = [x for x in stopwords.words('english') if x not in stop_word_to_keep]

    without_stopwords = [word for word in tokenized if not word in stop_words] # Remove Stop Words

    lemmatizer = WordNetLemmatizer() # Instantiate lemmatizer

    lemmatized = [lemmatizer.lemmatize(word) for word in without_stopwords] # Lemmatize

    lemmatized = " ".join(lemmatized)

    return lemmatized

# load model
model = models.load_model('models/models.h5')

# load tokenizer
with open('tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

airport_code_list = list(pd.read_csv('data/raw_data/airline_codes.csv')['ICAO'].dropna())

st.set_page_config(page_title="Model Demo",)

st.markdown("# Model Demo")


title = st.text_input('Tweet')


if st.button('click me'):
    # print is visible in the server output, not in the page

    print('button clicked!')

    X_pred = clean(title, airport_code_list)

    X_test_token = tokenizer.texts_to_sequences([X_pred])

    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post')

    y_pred = model.predict(X_test_pad)

    if y_pred[0] > 0.5:
        st.write('Negative tweet')
    else:
        st.write('Neutral or positive tweet')
