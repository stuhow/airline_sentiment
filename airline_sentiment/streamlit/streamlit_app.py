import streamlit as st
import pandas as pd
import pickle
from airline_sentiment.ml_logic.data import clean
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences


title = st.text_input('Tweet')


if st.button('click me'):
    # print is visible in the server output, not in the page

    print('button clicked!')

    airport_code_list = list(pd.read_csv('data/raw_data/airline_codes.csv')['ICAO'].dropna())

    X_pred = clean(title, airport_code_list)

    # load model
    model = models.load_model('models/models.h5')

    # load tokenizer
    with open('tokenizer/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    X_test_token = tokenizer.texts_to_sequences([X_pred])

    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post')

    y_pred = model.predict(X_test_pad)

    if y_pred[0] > 0.5:
        st.write('Negative tweet')
    else:
        st.write('Neutral or positive tweet')
