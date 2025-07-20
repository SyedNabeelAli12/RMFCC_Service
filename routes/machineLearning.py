import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, classification_report
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt



def load_data():
    economy_df = pd.read_csv('economy.csv')
    piracy_df = pd.read_csv('piracy.csv')
    return economy_df, piracy_df


def preprocess_economy(economy_df):
    columns_to_fill = ['GDP', 'INDUSTRYGDP', 'FISHPRODUCTION',
                       'MILITARY', 'CORRUPTIONINDEX', 'UNEMPLOYMENT']
    for col in columns_to_fill:
        economy_df[col] = economy_df[col].fillna(economy_df[col].median())

    economy_df = economy_df[economy_df['YEAROFRECORD'] >= 2000]
    economy_df['GDP_per_capita'] = economy_df['GDP'] / economy_df['POPULATION']
    economy_df['Fish_per_capita'] = economy_df['FISHPRODUCTION'] / economy_df['POPULATION']

    scaler = MinMaxScaler()
    economy_df[['MILITARY_scaled', 'UNEMPLOYMENT_scaled']] = scaler.fit_transform(
        economy_df[['MILITARY', 'UNEMPLOYMENT']])
    return economy_df, scaler


def aggregate_piracy(piracy_df):
    piracy_df['date'] = pd.to_datetime(piracy_df['date'], errors='coerce')
    piracy_df['YEAROFRECORD'] = piracy_df['date'].dt.year
    piracy_counts = piracy_df.groupby(['nearest_country', 'YEAROFRECORD']).size().reset_index(name='PIRACY_COUNT')
    piracy_counts.rename(columns={'nearest_country': 'COUNTRY'}, inplace=True)
    return piracy_counts


def merge_data(economy_df, piracy_counts):
    merged_df = pd.merge(economy_df, piracy_counts, on=['COUNTRY', 'YEAROFRECORD'], how='left')
    merged_df['PIRACY_COUNT'] = merged_df['PIRACY_COUNT'].fillna(0)
    return merged_df



def compute_features(df, scaler):
    econ_scaled = scaler.fit_transform(df[['GDP_per_capita', 'GR', 'INDUSTRYGDP']])
    df['Econ_Score'] = econ_scaled.mean(axis=1)
    risk_scale =scaler.fit_transform(df[['PIRACY_COUNT', 'CORRUPTIONINDEX']])
    df['Security_Risk'] = risk_scale.mean(axis=1)
    df['Fish_Score'] = scaler.fit_transform(df[['Fish_per_capita']])[:, 0]
    return df



def perform_clustering(df):
    features = df[['Econ_Score', 'Security_Risk', 'Fish_Score']]
    features_imputed = SimpleImputer(strategy='mean').fit_transform(features)
    best_k = 0
    best_score = -1
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(features_imputed)
        score = silhouette_score(features_imputed, labels)
        print(f"k={k} -> Silhouette Score: {score:.3f}")
        if score > best_score:
            best_k = k
            best_score = score
       

    kmeans = KMeans(n_clusters=best_k, random_state=0)
    df['Cluster'] = kmeans.fit_predict(features_imputed)

    silhouette = silhouette_score(features_imputed, df['Cluster'])
    print(f"📈 Silhouette Score: {silhouette:.3f}")

    db = DBSCAN(eps=0.3, min_samples=10).fit(features_imputed)
    df['DBSCAN_Outlier'] = db.labels_
    return df

def apply_pca(df):
    features = df[['Econ_Score', 'Security_Risk']]
    features_imputed = SimpleImputer(strategy='mean').fit_transform(features)
    pca = PCA(n_components=2)
    components = pca.fit_transform(features_imputed)
    df['PCA1'], df['PCA2'] = components[:, 0], components[:, 1]
    return df

def score_countries(df):
    df['Final_Score'] = (
        0.4 * df['Econ_Score'] +
        0.3 * (1 - df['Security_Risk']) +
        0.3 * df['Fish_Score']
    )
    return df


def forecast_trend(df, country, feature='GDP', forecast_years=5, plot=False):
 
    try:
        country_df = df[df['COUNTRY'] == country].sort_values('YEAROFRECORD')

        if feature not in country_df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset.")
        
        ts = country_df.set_index('YEAROFRECORD')[feature].dropna()

        if len(ts) < 5:
            print(f"⚠️ Not enough data to forecast for {country} and {feature}.")
            return None, None, None

        # Fit ARIMA model
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_years)
        future_years = np.arange(ts.index.max() + 1, ts.index.max() + 1 + forecast_years)

        forecast_df = pd.DataFrame({
            'YEAR': future_years,
            f'Forecast_{feature}': forecast.values,
            'Type': 'Forecast'
        })

        history_df = pd.DataFrame({
            'YEAR': ts.index,
            f'Forecast_{feature}': ts.values,
            'Type': 'Actual'
        })

        full_trend_df = pd.concat([history_df, forecast_df], ignore_index=True)

        last_actual = ts.iloc[-1]
        fifth_year_value = forecast.values[-1]
        percentage_change = ((fifth_year_value - last_actual) / last_actual) * 100

        year_summary = {
            'year': int(future_years[-1]),
            'country':country,
            'feature':feature,
            'forecast_value': float(fifth_year_value),
            'change_percent': float(percentage_change)
        }
        

        return full_trend_df, year_summary, forecast_df

    except Exception as e:
        print(f"❌ Error during forecasting for {country} - {feature}: {e}")
        return None, None, None


def forecast_and_predict_5th_year(df, country):
    features = ['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION']
    forecasts = {}
    forecast_values = {}
    year_summaries = {}

    for feature in features:
        full_trend_df, year_summary, forecast_df = forecast_trend(
            df, country, feature=feature, forecast_years=5, plot=False
        )

        if forecast_df is None:
            print(f"❌ Unable to forecast {feature} for {country}. Skipping prediction.")
            result ={
                "error": f"Not enough data or failed to forecast '{feature}' for {country}.",
                "feature_failed": feature
            }
            print(result)
            return result
        

        forecast_value = forecast_df.iloc[-1][f'Forecast_{feature}']
        forecasts[feature] = forecast_df
        forecast_values[feature] = forecast_value
        year_summaries[feature] = year_summary

    try:
        prediction_input = {
            'GR': forecast_values['GR'],
            'MILITARY': forecast_values['MILITARY'],
            'CORRUPTIONINDEX': forecast_values['CORRUPTIONINDEX'],
            'FISHPRODUCTION': forecast_values['FISHPRODUCTION']
        }

        prediction_result = predict(prediction_input, df)
        print( {
            "country": country,
            "5th_year_forecasts": forecast_values,
            "piracy_risk_prediction": prediction_result,
            "year_summary": year_summaries
        }
)

        return {
            "country": country,
            "5th_year_forecasts": forecast_values,
            "piracy_risk_prediction": prediction_result,
            "year_summary": year_summaries
        }

    except Exception as e:
        print(f"❌ Prediction failed for {country}: {e}")
        return {
            "error": f"Prediction step failed for {country}.",
            "exception": str(e)
        }



def predict(new_data, df):
    model_file = 'piracy_risk_model.joblib'
    imputer_file = 'piracy_imputer.joblib'
    scaler_file = 'piracy_scaler.joblib'

    if not (os.path.exists(model_file) and os.path.exists(imputer_file) and os.path.exists(scaler_file)):
        print("⚠️ Model files not found. Training new model...")
        

        df_train = df.dropna(subset=['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION'])
        df_train['High_Piracy'] = (df_train['PIRACY_COUNT'] > 0).astype(int)

        X = df_train[['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION']]
        y = df_train['High_Piracy']

        imputer = SimpleImputer(strategy='mean')
        scaler = MinMaxScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight='balanced', random_state=42
        )
        model.fit(X_train, y_train)

        joblib.dump(model, model_file)
        joblib.dump(imputer, imputer_file)
        joblib.dump(scaler, scaler_file)

        y_pred = model.predict(X_test)
        print("✅ Model trained. Classification Report:\n", classification_report(y_test, y_pred))

        print("Feature importances:")
        for name, importance in zip(['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION'], model.feature_importances_):
            print(f"{name}: {importance:.3f}")

        print("Feature importances:")
        for name, importance in zip(['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION'], model.feature_importances_):
            print(f"{name}: {importance:.3f}")

    else:
        model = joblib.load(model_file)
        imputer = joblib.load(imputer_file)
        scaler = joblib.load(scaler_file)

        print("✅ Loaded existing model and transformers.")
        print("Feature importances:")
        for name, importance in zip(['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION'], model.feature_importances_):
            print(f"{name}: {importance:.3f}")

    input_df = pd.DataFrame([new_data])
    X_input_imputed = imputer.transform(input_df)
    X_input_scaled = scaler.transform(X_input_imputed)

    prediction = model.predict(X_input_scaled)
    prediction_proba = model.predict_proba(X_input_scaled)

    return {
        'High_Piracy': int(prediction[0]),
        'Probability_No': prediction_proba[0][0],
        'Probability_Yes': prediction_proba[0][1]
    }
    