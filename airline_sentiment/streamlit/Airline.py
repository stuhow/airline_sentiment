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
# client = bigquery.Client(credentials=credentials)

st.set_page_config(
    page_title="Airline Dashboard",
    page_icon="👋",
)

st.write("# Welcome to my airline dashboard")
st.write("### See how an airlines twitter sentiment changes over time and the reasons behind the change")
st.write("#### Select a time frame and a airline from the below drop downs and then click me!")


option = st.selectbox('Select a time frame', ['Day', 'Week', 'Month', 'Year'])

@st.cache_data
def get_bar_chart_data(directory):
    ''' function to load data locally'''
    df = pd.read_csv(directory,
                    lineterminator='\n')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def get_bq_chart_data(airline_option):
    ''' function to load data from bigquery'''
    client = bigquery.Client(credentials=credentials)

    sql = f"""
        SELECT date, pred, topic_customer_service, topic_flight
        FROM `{project}.{dataset}.{airline_option}_predictions`
    """

    df = client.query(sql).to_dataframe()

    return df


airline = st.selectbox('Select an airline', ['United Airlines', 'American Airlines'], key=1)

airline_dict = {'United Airlines': 'united', 'American Airlines': 'AmericanAir'}

# airline_option1 = airline_dict[airline]

# directory = f'data/predicted_data/{airline_option1}/{airline_option1}_predictions.csv'

date_band_dict = {'Day': 'D', 'Week': 'W', 'Month': 'M', 'Year': 'Y'}


if st.button('click me', key=3):
    # print is visible in the server output, not in the page

    print('button clicked!')

    # load dataframe
    airline_option1 = airline_dict[airline]
    # chart_data = get_bar_chart_data(directory)
    chart_data = get_bq_chart_data(airline_option1)

    chart_data['date'] = pd.to_datetime(chart_data['date'])

    chart_data = chart_data.set_index('date')

    # show overall sentiment
    chart_data_overall = chart_data.groupby(pd.Grouper(freq=date_band_dict[option]))['pred'].mean().reset_index()
    chart_data_overall['date'] = chart_data_overall['date'].dt.strftime('%Y-%m-%d')

    chart_data_overall = chart_data_overall.set_index('date')

    chart_data_overall['pred'] = chart_data_overall['pred'] * 100

    chart_data_overall = chart_data_overall.rename({'pred': 'Negative tweets'}, axis=1)

    st.write(f'Percentage of {airline} tweets that are negative')
    st.bar_chart(chart_data_overall)


    # show % of neg tweets by Customer service issue or flight issue

    chart_data_prob = pd.DataFrame(chart_data.groupby(pd.Grouper(freq=date_band_dict[option]))['pred'].mean())

    for i in ['topic_customer_service','topic_flight']:

        chart_data_topic_option = chart_data[chart_data['pred'] == 1]
        chart_data_topic_option = pd.DataFrame(chart_data_topic_option.groupby(pd.Grouper(freq=date_band_dict[option]))[i].mean())

        merged_df = chart_data_prob.join(chart_data_topic_option)

        merged_df_final = merged_df.drop(columns=['pred'])

        merged_df_final[i] = merged_df_final[i] * 100

        merged_df_final = merged_df_final.rename({i: f'{i[6:].replace("_", " ")}'}, axis=1)

        st.write(f'Percentage of negative tweets related to the {i[6:].replace("_", " ")}')
        st.bar_chart(merged_df_final)
