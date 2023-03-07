from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

@app.get('/predict')
def predict():
    return {'wait': 64}
