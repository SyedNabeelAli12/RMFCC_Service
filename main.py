from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

from fastapi.middleware.cors import CORSMiddleware



# 👇 Import your existing functions here
from ml import (
    load_data, preprocess_economy, aggregate_piracy, merge_data,
    compute_features, perform_clustering, apply_pca,
    score_countries, predict_piracy
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your React app’s domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store pipeline data globally
model = None
merged_df = None
top_countries = None

class CountryInput(BaseModel):
    GR: float
    MILITARY: float
    CORRUPTIONINDEX: float
    FISHPRODUCTION: float

@app.on_event("startup")
def startup_event():
    global model, merged_df, top_countries

    print("🔄 Running data pipeline...")
    economy_df, piracy_df = load_data()
    economy_df, scaler = preprocess_economy(economy_df)
    piracy_counts = aggregate_piracy(piracy_df)
    merged_df = merge_data(economy_df, piracy_counts)
    merged_df = compute_features(merged_df, scaler)
    merged_df = perform_clustering(merged_df)
    merged_df = apply_pca(merged_df)
    merged_df = score_countries(merged_df)

    # 🏆 Compute top countries
    top_countries = (
        merged_df.groupby('COUNTRY')['Final_Score']
        .mean()
        .sort_values(ascending=False)
        .head(15)
    ).reset_index().to_dict(orient='records')

    print("\n🏆 Top Recommended Countries for Expansion:\n")
    for country in top_countries:
        print(country)

    # Train or load model
    predict_piracy(merged_df)

    # Load model for prediction
    model_file = 'piracy_risk_model.joblib'
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print(f"✅ Loaded model: {model_file}")
    else:
        raise FileNotFoundError("Model file not found. Please check pipeline.")

@app.get("/")
def read_root():
    return {
        "message": "🚢 Welcome to the Piracy Risk API!",
        "top_countries": top_countries
    }

@app.post("/predict")
def predict_country(input_data: CountryInput):
    global model

    new_data = np.array([[input_data.GR, input_data.MILITARY, input_data.CORRUPTIONINDEX, input_data.FISHPRODUCTION]])

    prediction = model.predict(new_data)[0]

    risk_label = "High Piracy Risk" if prediction == 1 else "Low Piracy Risk"
    recommendation = (
        "🎉 This country is a good candidate for expansion!"
        if prediction == 0 else
        "⚠️ This country may have high piracy risk. Be cautious."
    )

    return {
        "prediction": risk_label,
        "recommendation": recommendation
    }
