import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery

project = st.secrets["PROJECT"]
dataset = st.secrets["DATASET"]

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)


st.set_page_config(
    page_title="Airline Dashboard",
    page_icon="👋",
)

st.write("## Compare the twitter sentiment of the 2 biggest US airlines!")
st.write("### Select a time frame from the drop down and click me!")

option = st.selectbox('Select a time frame', ['Day', 'Week', 'Month', 'Year'])

@st.cache_data
def get_bar_chart_data(directory):
    df = pd.read_csv(directory,
                    lineterminator='\n')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def get_bq_chart_data(airline_option1):
    ''' function to load data from bigquery'''
    client = bigquery.Client(credentials=credentials)

    sql = f"""
        SELECT date, pred, topic_customer_service, topic_flight
        FROM `{project}.{dataset}.{airline_option1}_predictions`
    """

    df = client.query(sql).to_dataframe()

    return df

# airline_option1 = st.selectbox('### Select an airline', ['AmericanAir', 'united', 'JetBlue'], key=1)
airline_option1 = 'AmericanAir'

directory = f'data/predicted_data/{airline_option1}/{airline_option1}_predictions.csv'

# airline_option2 = st.selectbox('### Select an airline', ['united', 'AmericanAir', 'JetBlue'], key=2)

airline_option2 = 'united'

directory2 = f'data/predicted_data/{airline_option2}/{airline_option2}_predictions.csv'


col1, col2 = st.columns(2)

if st.button('click me', key=3):
    # print is visible in the server output, not in the page

    print('button clicked!')

    with col1:
            # chart_data = get_bar_chart_data(directory)

            chart_data = get_bq_chart_data(airline_option1)

            chart_data['date'] = pd.to_datetime(chart_data['date'])

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

            chart_data['pred'] = chart_data['pred'] * 100

            chart_data = chart_data.rename({'pred': 'Negative tweets'}, axis=1)

            chart_data = chart_data.set_index('date')
            st.write(f'Percentage of American Airlines tweets that are negative')
            st.bar_chart(chart_data)

    with col2:
        # chart_data = get_bar_chart_data(directory2)

        chart_data = get_bq_chart_data(airline_option2)

        chart_data['date'] = pd.to_datetime(chart_data['date'])

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

        chart_data['pred'] = chart_data['pred'] * 100

        chart_data = chart_data.rename({'pred': 'Negative tweets'}, axis=1)

        chart_data = chart_data.set_index('date')
        st.write(f'Percentage of United Airlines tweets that are negative')
        st.bar_chart(chart_data)
