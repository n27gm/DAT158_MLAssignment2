import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from car_features import add_features, to_model_frame

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")
MODEL_DIR = ROOT_DIR / "model"

model = joblib.load(MODEL_DIR / "lgbm_model.joblib")
categories = json.loads((MODEL_DIR / "categories.json").read_text())
options = json.loads((MODEL_DIR / "options.json").read_text())
model_metrics = json.loads((MODEL_DIR / "metrics.json").read_text())

app = FastAPI(title="Bruktbilpris API")
api = APIRouter(prefix="/api")


class CarInput(BaseModel):
    brand: str
    model: str
    model_year: int = Field(ge=1970, le=2024)
    milage: float = Field(ge=0, le=1_000_000)
    fuel_type: str
    engine: str
    transmission: str
    ext_col: str
    int_col: str
    accident: str
    clean_title: str


@api.get("/")
def root():
    return {"message": "Bruktbilpris API kjører"}


@api.get("/options")
def get_options():
    return {**options, "metrics": model_metrics}


@api.post("/predict")
def predict(car: CarInput):
    df = add_features(pd.DataFrame([car.model_dump()]))
    price = float(model.predict(to_model_frame(df, categories))[0])
    return {"price": round(max(price, 0), 0)}


app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"], #os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
