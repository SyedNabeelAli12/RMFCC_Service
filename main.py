from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

from fastapi.middleware.cors import CORSMiddleware

# 👇 Import your pipeline functions
from ml import (
    load_data, preprocess_economy, aggregate_piracy, merge_data,
    compute_features, perform_clustering, apply_pca,
    score_countries, predict
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store globals
model = None
imputer = None
scaler = None
top_countries = None


class CountryInput(BaseModel):
    GR: float
    MILITARY: float
    CORRUPTIONINDEX: float
    FISHPRODUCTION: float


@app.on_event("startup")
def startup_event():
    global model, imputer, scaler, top_countries, merged_df

    print("🔄 Running data pipeline...")

    economy_df, piracy_df = load_data()
    economy_df, econ_scaler = preprocess_economy(economy_df)
    piracy_counts = aggregate_piracy(piracy_df)
    merged_df = merge_data(economy_df, piracy_counts)
    merged_df = compute_features(merged_df, econ_scaler)
    merged_df = perform_clustering(merged_df)
    merged_df = apply_pca(merged_df)
    merged_df = score_countries(merged_df)

    # Compute top countries
    top_countries = (
        merged_df.groupby('COUNTRY')['Final_Score']
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
        .to_dict(orient='records')
    )
    print("\n🏆 Top Recommended Countries:\n")
    for country in top_countries:
        print(country)

    # Train or load model


@app.get("/")
def read_root():
    return {
        "message": "🚢 Welcome to the Piracy Risk API!",
        "top_countries": top_countries
    }

@app.post("/predict")
def predict_country(input_data: CountryInput):
    global merged_df

    # Build dict for the predict function
    new_data = {
        "GR": input_data.GR,
        "MILITARY": input_data.MILITARY,
        "CORRUPTIONINDEX": input_data.CORRUPTIONINDEX,
        "FISHPRODUCTION": input_data.FISHPRODUCTION
    }

    print(f"📥 New input: {new_data}")

    # Call your robust predict function
    prediction_result = predict(new_data, merged_df)
    print(prediction_result)

    return prediction_result