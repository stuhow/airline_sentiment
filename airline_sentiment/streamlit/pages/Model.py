import streamlit as st
import requests

st.write("## Test out the model predictions with its API")
st.write("### Enter a tweet below and click on Make prediction!")

with st.form(key='params_for_api'):
    example_text = st.text_input("Enter some text 👇", "the worst flight i've ever taken")
    submit_button = st.form_submit_button('Make prediction')

if submit_button:
    params = dict(text=example_text)
    api_url = 'https://sentimentapi-ptw3tzx4aq-ew.a.run.app/predict'
    response = requests.get(api_url, params=params)
    prediction = response.json()
    pred = prediction['prediction']
    st.header(f'This is a {pred}')
