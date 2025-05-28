from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Serve static files (HTML page)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("templet\index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # Ensure the input order matches the training order
    features = [
        data["len"],
        data["numdigit"],
        data["has_ip"],
        data["_https"],
        data["Scharacter"],
        data["@_symbol"],
        data["http_"],
        data["subdomains"],
        data["domain_length"],
        data["entropy"]
    ]

    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    return JSONResponse({"prediction": int(prediction[0])})
