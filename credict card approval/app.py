import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    features = [float(x) for x in form.values()]
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    result = 'Approved' if prediction[0] == 1 else 'Not Approved'
    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    