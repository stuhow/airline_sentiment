FROM tensorflow/tensorflow
# FROM python:3.8.6-buster

COPY airline_sentiment /airline_sentiment
COPY data/raw_data/airline_codes.csv /data/raw_data/airline_codes.csv
COPY requirements_api.txt /requirements.txt
COPY models /models
COPY tokenizer /tokenizer

ENV NLTK_DATA /nltk_data/ ADD . $NLTK_DATA

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt -d /usr/share/nltk_data
RUN python -m nltk.downloader stopwords -d /usr/share/nltk_data
RUN python -m nltk.downloader wordnet -d /usr/share/nltk_data

CMD PYTHONPATH=airline_sentiment/api/ uvicorn fast:app --host 0.0.0.0 --port $PORT
