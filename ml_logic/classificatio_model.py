from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

def embedding(X_train):
    ''''
    function that takes the cleaned text, trains a tokenizer and fits the
    tokenizer to the traing data.
    It also saves the fitted tokenizer to be used on the test and predict
    functions later

    '''
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(X_train)

    X_train_token = tokenizer.texts_to_sequences(X_train)

    # save tokenizer
    with open('tokenizer/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #check vocab length
    vocab_size = len(tokenizer.word_index)

    # pad sequences
    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post')

    return X_train_pad, vocab_size # X_test_pad,


def initialize_model(vocab_size, embedding_dimension = 100):
    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size + 1,
                            output_dim=embedding_dimension,
                            mask_zero=True))

    # model.add(layers.BatchNormalization())
    model.add(layers.LSTM(30, activation="tanh",return_sequences=True))
    model.add(layers.LSTM(30, activation="tanh"))
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Dense(1, activation="sigmoid"))

    return model

def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def train_model(model,
                X_train_pad,
                y_train):

    es = EarlyStopping(patience=4, restore_best_weights=True)

    history = model.fit(X_train_pad, y_train,
                epochs=50,
                batch_size=16,
                validation_split=0.3,
                callbacks=[es]
                )

    return model, history

def evaluate_model(model,
                   X,
                   y):

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=32,
        verbose=1,
        return_dict=True)

    return metrics
