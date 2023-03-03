import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unidecode
import contractions
from sklearn.utils import resample

def get_airline_codes():
    airport_code_list = list(pd.read_html('https://en.wikipedia.org/wiki/List_of_airline_codes')[0]['ICAO'].dropna())
    return airport_code_list


def clean(text, airport_code_list):

    for air_code in airport_code_list: # remove airline codes
        text = text.replace(air_code, ' ')

    text = text.lower() # Lower Case

    text = re.sub("@[A-Za-z0-9_]+","", text) # remove mentions
    text = re.sub("#[A-Za-z0-9_]+","", text) # remove hashtags

    text = re.sub(r"http\S+", "", text) # remove links
    text = re.sub(r"www.\S+", "", text) # remove links

    expanded_words = [contractions.fix(word) for word in text.split()] # remove contractions

    text = ' '.join(expanded_words) # join words

    unaccented_string = unidecode.unidecode(text) # remove accents

    tokenized = word_tokenize(unaccented_string) # Tokenize

    stop_words = set(stopwords.words('english')) # Make stopword list

    stop_word_to_keep = ['was', 'are', 'did', 'been', 'have', 'until', 'while', 'about', 'against', 'between', 'during', 'before', 'after', 'again', 'when', 'where', 'why', 'how', 'any', 'not', 'no','very', "aren't", "wasn't", "shouldn't", "should", "won't", "wouldn't"]

    stop_words = [x for x in stopwords.words('english') if x not in stop_word_to_keep]

    without_stopwords = [word for word in tokenized if not word in stop_words] # Remove Stop Words

    lemmatizer = WordNetLemmatizer() # Instantiate lemmatizer

    lemmatized = [lemmatizer.lemmatize(word) for word in without_stopwords] # Lemmatize

    lemmatized = " ".join(lemmatized)

    return lemmatized

def balance_training_df(X_train, y_train):
    # create training DF
    train_df = pd.DataFrame(X_train)
    train_df['airline_sentiment'] = y_train

    # seperate negative and neutral tweets
    negative_messages = train_df[train_df['airline_sentiment'] == 1]
    neutral_messages  = train_df[train_df['airline_sentiment'] == 0]

    # down sample tweets
    negative_downsample = resample(negative_messages,
                replace=True,
                n_samples=len(neutral_messages),
                random_state=42)

    # combine resampled datasets
    train_df = pd.concat([negative_downsample, neutral_messages])

    return train_df
