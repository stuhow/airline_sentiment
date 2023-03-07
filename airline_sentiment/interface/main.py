from airline_sentiment.data_sources.local_disk import load_raw_data_local, save_clean_data_local
from airline_sentiment.ml_logic.data import get_airline_codes, clean, balance_training_df
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pickle
from airline_sentiment.ml_logic.classificatio_model import evaluate_model
from airline_sentiment.data_sources.local_disk import load_clean_tweets
from airline_sentiment.ml_logic.classificatio_model import initialize_model, compile_model, train_model, embedding
from tensorflow.keras import models



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

    # load dataset
    df = load_clean_tweets(source = 'train')

    # set X & y
    X = df['text'].astype("str")
    y =  df [['airline_sentiment']]

    # embed
    X_pad, vocab_size = embedding(X)

    # initialize model
    model = initialize_model(vocab_size, embedding_dimension = 100)

    #compile model
    model = compile_model(model)

    # fit model
    model, history = train_model(
            model,
            X_pad,
            y)

    # save model
    model.save('models/models.h5')

    return None

def pred():

    X_pred = input('Predict sentiment of: ')

    airport_code_list = get_airline_codes()

    X_pred = clean(X_pred, airport_code_list)

    # load model
    model = models.load_model('models/models.h5')

    # load tokenizer
    with open('tokenizer/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    X_test_token = tokenizer.texts_to_sequences([X_pred])

    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post')

    y_pred = model.predict(X_test_pad)

    if y_pred[0] == 1:
        print('Negative')
    else:
        print('Neutral or positive')
    return y_pred

def evaluate():

    # load data from clean data
    df = load_clean_tweets(source = 'test')

    # set variables
    X_test = df['text'].astype("str")
    y_test = df[['airline_sentiment']]

    # load model
    model = models.load_model('models/models.h5')

    # loading
    with open('tokenizer/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    X_test_token = tokenizer.texts_to_sequences(X_test)

    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post')

    metrics = evaluate_model(model, X_test_pad, y_test)

    return metrics


if __name__ == '__main__':
    preprocess()
    train()
    # pred()
    evaluate()
