FROM tensorflow/tensorflow:2.10.0

COPY airline_sentiment /airline_sentiment
COPY airline_sentiment/data/raw_data/airline_codes data/raw_data/airline_codes
COPY requirements.txt requirements.txt

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# CMD PYTHONPATH=airline_sentiment/api/ uvicorn fast:app --reload
