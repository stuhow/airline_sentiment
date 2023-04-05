FROM tensorflow/tensorflow
# FROM python:3.8.6-buster

COPY airline_sentiment /airline_sentiment
COPY data/raw_data/airline_codes.csv /data/raw_data/airline_codes.csv
COPY requirements_api.txt /requirements.txt
COPY models /models
COPY tokenizer /tokenizer

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

CMD PYTHONPATH=airline_sentiment/api/ uvicorn fast:app --host 0.0.0.0 --port $PORT
