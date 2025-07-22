from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from routes.machineLearning import (
    load_data, preprocess_economy, aggregate_piracy, merge_data,
    compute_features, perform_clustering, apply_pca,
    score_countries, predict, forecast_trend, forecast_and_predict_5th_year
)
from middleware.middleware import(verify_token)
from routes.user import router as user_router 
from pydantic import BaseModel
from dotenv import load_dotenv
import os


load_dotenv()


# Init app
app = FastAPI()
origins = os.getenv("URL", "http://localhost:3000").split(",")
# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # ⚠️ Allows any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"]
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

class ForecastInput(BaseModel):
    country: str
    feature: str 
    years: int 
class Forecast5YearInput(BaseModel):
    country: str

@app.on_event("startup")
def startup_event():
    global model, imputer, scaler, top_countries, merged_df

    print("Running data pipeline...")
    economy_df, piracy_df = load_data()
    economy_df, econ_scaler = preprocess_economy(economy_df)
    piracy_counts = aggregate_piracy(piracy_df)
    merged_df = merge_data(economy_df, piracy_counts)
    merged_df = compute_features(merged_df, econ_scaler)
    merged_df = perform_clustering(merged_df)
    merged_df = apply_pca(merged_df)
    merged_df = score_countries(merged_df)

    top_countries = (
        merged_df.groupby(['COUNTRY','COUNTRYNAME'])['Final_Score']
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
def read_root(user=Depends(verify_token)):
    return {
        "message": "🚢 Welcome to the Piracy Risk API!",
        "top_countries": top_countries
    }

@app.post("/forecast")
def forecast_post(input_data: ForecastInput,user=Depends(verify_token)):
    global merged_df

    full_df, year5_info,forecast_df  = forecast_trend(
        merged_df,
        country=input_data.country,
        feature=input_data.feature,
        forecast_years=input_data.years,
        plot=False
    )

    if forecast_df is None:
        return {
            "error": f"⚠️ Not enough data to forecast for '{input_data.country}' with feature '{input_data.feature}'."
        }

    return {
        "country": input_data.country,
        "feature": input_data.feature,
        "forecast": forecast_df.to_dict(orient="records"),
        "5th_year_summary": year5_info,
        "full_df" :full_df.to_dict(orient="records")
    }


@app.post("/forecast-predict")
def forecast_and_predict_route(input_data: Forecast5YearInput,user=Depends(verify_token)):
    global merged_df

    result = forecast_and_predict_5th_year(merged_df, input_data.country.upper())
    return result

@app.get("/distinct-countries")
def get_distinct_countries(user=Depends(verify_token)):
    global merged_df
   
    distinct_countries = (
        merged_df[["COUNTRY", "COUNTRYNAME"]]
        .dropna()
        .drop_duplicates()
        .sort_values("COUNTRYNAME")
        .to_dict(orient="records")
    )
    return {"countries": distinct_countries}

@app.post("/predict")
def predict_country(input_data: CountryInput,user=Depends(verify_token)):
    global merged_df
    new_data = {
        "GR": input_data.GR,
        "MILITARY": input_data.MILITARY,
        "CORRUPTIONINDEX": input_data.CORRUPTIONINDEX,
        "FISHPRODUCTION": input_data.FISHPRODUCTION
    }
    prediction_result = predict(new_data, merged_df)
    return prediction_result
