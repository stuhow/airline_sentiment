from data_sources.local_disk import load_raw_data_local, save_clean_data_local
from ml_logic.data import get_airline_codes, clean, balance_training_df
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

def preprocess()-> pd.DataFrame:
    ''''clean data and save it'''
    # import data
    raw_df = load_raw_data_local()

    print(f"\n✅ Data loaded")

    # set X & y
    X = raw_df['text']
    y = raw_df[['airline_sentiment']]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # get the list of airline codes
    airport_code_list = get_airline_codes()

    print(f"\n✅ Airline code retrieved")

    # clean tweets
    X_train = [clean(tweet, airport_code_list) for tweet in X_train]
    X_test = [clean(tweet, airport_code_list) for tweet in X_test]

    print(f"\n✅ Text cleaned")

    # change sentiment from positive to neutral
    y_train = y_train.replace({'positive':'neutral'})
    y_test = y_test.replace({'positive':'neutral'})

    # fit odrinal encoder for target variable
    enc = OrdinalEncoder(categories=[['neutral','negative']])
    enc.fit(y_train)

    # transform y train and test
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    print(f"\n✅ Target & test transformed")

    #balance training set, renaming columns and export
    train_df = balance_training_df(X_train, y_train)

    train_df = train_df.rename(columns={0:'text'})

    save_clean_data_local(train_df, target='train')

    # create testing DF
    test_df = pd.DataFrame(X_test)
    test_df['airline_sentiment'] = y_test
    test_df = test_df.rename(columns={0:'text'})
    save_clean_data_local(test_df, target='test')

    print(f"\n✅ Data processed and saved")

    return None

def train():
    pass

def pred():
    pass

def evaluate():
    pass


if __name__ == '__main__':
    preprocess()
    # train()
    # pred()
    # evaluate()
