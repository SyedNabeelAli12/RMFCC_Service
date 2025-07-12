from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.ml import (
    load_data, preprocess_economy, aggregate_piracy, merge_data,
    compute_features, perform_clustering, apply_pca,
    score_countries, predict
)
from routes.user import router as user_router 

from pydantic import BaseModel

# Init app
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Allows any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Register user router
app.include_router(user_router)

# Globals
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


@app.get("/")
def read_root():
    return {
        "message": "🚢 Welcome to the Piracy Risk API!",
        "top_countries": top_countries
    }


@app.post("/predict")
def predict_country(input_data: CountryInput):
    global merged_df
    new_data = {
        "GR": input_data.GR,
        "MILITARY": input_data.MILITARY,
        "CORRUPTIONINDEX": input_data.CORRUPTIONINDEX,
        "FISHPRODUCTION": input_data.FISHPRODUCTION
    }
    prediction_result = predict(new_data, merged_df)
    return prediction_result
