import pandas as pd


def load_raw_data_local():
    df = pd.read_csv('data/raw_data/Tweets.csv')
    return df

def save_clean_data_local(df, target='train'):
    df.to_csv(f'data/clean_data/clean_tweets_{target}.csv', index=False)
    return None

def load_clean_tweets(source = 'train'):
    df = pd.read_csv(f'data/clean_data/clean_tweets_{source}.csv')
    return df
