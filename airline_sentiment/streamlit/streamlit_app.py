import streamlit as st
import pandas as pd
import pickle
from airline_sentiment.ml_logic.data import clean
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences




st.set_page_config(
    page_title="Airline Dashboard",
    page_icon="👋",
)

st.write("# Welcome to my airline dashboard")


option = st.selectbox('### Select a date band', ['Day', 'Week', 'Month', 'Year'])

@st.cache_data
def get_bar_chart_data(directory):
    df = pd.read_csv(directory,
                    lineterminator='\n')
    df['date'] = pd.to_datetime(df['date'])
    return df

airline_option1 = st.selectbox('### Select an airline', ['AmericanAir', 'united', 'USAirways', 'SouthwestAir', 'JetBlue'], key=1)

directory = f'data/predicted_data/{airline_option1}/{airline_option1}_predictions.csv'

airline_option2 = st.selectbox('### Select an airline', ['united', 'USAirways', 'SouthwestAir', 'JetBlue', 'AmericanAir'], key=2)

directory2 = f'data/predicted_data/{airline_option2}/{airline_option2}_predictions.csv'


col1, col2 = st.columns(2)

if st.button('click me', key=3):
    # print is visible in the server output, not in the page

    print('button clicked!')

    with col1:
            chart_data = get_bar_chart_data(directory)

            chart_data = chart_data.set_index('date')

            if option == 'Day':
                chart_data = chart_data.groupby(pd.Grouper(freq="D"))['pred'].mean().reset_index()
                chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d')

            if option == 'Week':
                chart_data = chart_data.groupby(pd.Grouper(freq="W"))['pred'].mean().reset_index()
                chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d')

            if option == 'Month':
                chart_data = chart_data.groupby(pd.Grouper(freq="M"))['pred'].mean().reset_index()
                chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d').str[:7]

            if option == 'Year':
                chart_data = chart_data.groupby(pd.Grouper(freq="Y"))['pred'].mean().reset_index()
                chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d').str[:4]


            chart_data = chart_data.set_index('date')
            st.write(airline_option1)
            st.bar_chart(chart_data)

    with col2:
        chart_data = get_bar_chart_data(directory2)

        chart_data = chart_data.set_index('date')

        if option == 'Day':
            chart_data = chart_data.groupby(pd.Grouper(freq="D"))['pred'].mean().reset_index()
            chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d')

        if option == 'Week':
            chart_data = chart_data.groupby(pd.Grouper(freq="W"))['pred'].mean().reset_index()
            chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d')

        if option == 'Month':
            chart_data = chart_data.groupby(pd.Grouper(freq="M"))['pred'].mean().reset_index()
            chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d').str[:7]

        if option == 'Year':
            chart_data = chart_data.groupby(pd.Grouper(freq="Y"))['pred'].mean().reset_index()
            chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d').str[:4]


        chart_data = chart_data.set_index('date')
        st.write(airline_option2)
        st.bar_chart(chart_data)
