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

airline_option1 = st.selectbox('### Select an airline', ['SouthwestAir', 'AmericanAir', 'united', 'JetBlue'], key=1)

directory = f'data/predicted_data/{airline_option1}/{airline_option1}_predictions.csv'

date_band_dict = {'Day': 'D', 'Week': 'W', 'Month': 'M', 'Year': 'Y'}


if st.button('click me', key=3):
    # print is visible in the server output, not in the page

    print('button clicked!')

    # load dataframe

    chart_data = get_bar_chart_data(directory)

    chart_data = chart_data.set_index('date')

    # show overall sentiment
    chart_data_overall = chart_data.groupby(pd.Grouper(freq=date_band_dict[option]))['pred'].mean().reset_index()
    chart_data_overall['date'] = chart_data_overall['date'].dt.strftime('%Y-%m-%d')

    chart_data_overall = chart_data_overall.set_index('date')
    st.write(airline_option1)
    st.bar_chart(chart_data_overall)


    # show % of neg tweets by Customer service issue or flight issue

    chart_data_prob = chart_data.copy()
    chart_data_topic = chart_data.copy()

    chart_data_prob = pd.DataFrame(chart_data_prob.groupby(pd.Grouper(freq=date_band_dict[option]))['pred'].mean())

    for i in ['topic_customer service','topic_flight']:

        chart_data_topic_option = chart_data_topic[chart_data_topic['pred'] == 1]
        chart_data_topic_option = pd.DataFrame(chart_data_topic_option.groupby(pd.Grouper(freq=date_band_dict[option]))[i].mean())

        merged_df = chart_data_prob.join(chart_data_topic_option)

        merged_df_final = merged_df.copy()

        merged_df_final = merged_df_final.drop(columns=['pred'])

        st.write(airline_option1)
        st.bar_chart(merged_df_final)
