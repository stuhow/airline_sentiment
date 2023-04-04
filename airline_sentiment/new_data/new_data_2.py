temp_file_model = datapath("/home/stuart/code/stuhow/airline_sentiment/models/topic_model")

from gensim import  models
from gensim.test.utils import get_tmpfile

lda = models.ldamodel.LdaModel.load(temp_file_model)

tmp_fname = get_tmpfile("/home/stuart/code/stuhow/airline_sentiment/models/topic_model.id2word")
loaded_dct = Dictionary.load(tmp_fname)

df = pd.read_csv(f'data/predicted_data/JetBlue/JetBlue_predictions.csv', lineterminator='\n')

def topic_clean(text):
    for punctuation in string.punctuation:
        text = str(text).replace(punctuation, ' ') # Remove Punctuation

    text = text.replace('canceled', 'cancelled')

    tokenized = word_tokenize(text) # Tokenize

    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers

    stop_words = set(stopwords.words('english')) # Make stopword list

    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words

    without_short_words = [word for word in without_stopwords if len(word) > 2]

    without_flight = [word for word in without_short_words if word != 'flight']

    # without_stopwords = " ".join(without_stopwords)

    return without_flight



df['clean_topic_text'] = df['clean_text'].apply(topic_clean)


def topic_predictions(new_doc):
    new_doc_bow = loaded_dct.doc2bow(new_doc)
    return lda.get_document_topics(new_doc_bow)


df['topic_prediction'] = df['clean_topic_text'].apply(topic_predictions)


def topic_string(text):
    prob_list = [i[1] for i in text]
    topic_index = prob_list.index(max(prob_list))
    topic_dict = {0: 'flight', 1:'other', 2:'customer service'}
    return topic_dict[topic_index]

df['topic'] = df['topic_prediction'].apply(topic_string)

df = pd.get_dummies(df, columns = ['topic'])
