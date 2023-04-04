import snscrape.modules.twitter as sntwitter
import pandas as pd
from airline_sentiment.ml_logic.data import get_airline_codes, clean
from tensorflow.keras import models
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

airlines = ['AmericanAir', 'united', 'SouthwestAir', 'JetBlue']


airport_code_list = get_airline_codes()

# load model
model = models.load_model('models/models.h5')

# load tokenizer
with open('tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

for airline in airlines:
    tweets = []

    query = f'(@{airline}) lang:en until:2023-01-05 since:2022-01-01 -filter:replies'

    print(f'Starting tweet collection for {airline}')

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.content])

    if len(tweets) > 0:

        df = pd.DataFrame(tweets, columns=['date', 'text'])

        df['date'] = pd.to_datetime(df['date']).dt.date
        X = df['text'].astype("str")

        X_clean = [clean(tweet, airport_code_list) for tweet in X]

        X_token = tokenizer.texts_to_sequences(X_clean)

        X_pad = pad_sequences(X_token, dtype='float32', padding='post')

        y_pred = model.predict(X_pad)

        df['clean_text'] = X_clean
        df['pred'] = [1 if i[0] > 0.5 else 0 for i in y_pred]

        df.to_csv(f'data/predicted_data/{airline}/{airline}_predictions.csv', index=False)

        print(f'{airline} dataframe saved')

    else:
        print(f'{airline} has no tweets')
