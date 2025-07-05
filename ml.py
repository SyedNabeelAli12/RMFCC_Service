import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib
import os

# 📥 Data Loading
def load_data():
    economy_df = pd.read_csv('economy.csv')
    piracy_df = pd.read_csv('piracy.csv')
    return economy_df, piracy_df


# # 🧹 Preprocessing Economy Data
def preprocess_economy(economy_df):
    columns_to_fill = ['GDP', 'INDUSTRYGDP', 'FISHPRODUCTION', 'MILITARY', 'CORRUPTIONINDEX', 'UNEMPLOYMENT']
    for col in columns_to_fill:
        economy_df[col] = economy_df[col].fillna(economy_df[col].median())

    economy_df = economy_df[economy_df['YEAROFRECORD'] >= 2000]
    economy_df['GDP_per_capita'] = economy_df['GDP'] / economy_df['POPULATION']
    economy_df['Fish_per_capita'] = economy_df['FISHPRODUCTION'] / economy_df['POPULATION']

    scaler = MinMaxScaler()
    economy_df[['MILITARY_scaled', 'UNEMPLOYMENT_scaled']] = scaler.fit_transform(
        economy_df[['MILITARY', 'UNEMPLOYMENT']])
    return economy_df, scaler



# # 📊 Aggregating Piracy Data
def aggregate_piracy(piracy_df):
    piracy_df['date'] = pd.to_datetime(piracy_df['date'], errors='coerce')
    piracy_df['YEAROFRECORD'] = piracy_df['date'].dt.year
    piracy_counts = piracy_df.groupby(['nearest_country', 'YEAROFRECORD']).size().reset_index(name='PIRACY_COUNT')
    piracy_counts.rename(columns={'nearest_country': 'COUNTRY'}, inplace=True)
    return piracy_counts



# 🔀 Merging Datasets
def merge_data(economy_df, piracy_counts):
    merged_df = pd.merge(economy_df, piracy_counts, on=['COUNTRY', 'YEAROFRECORD'], how='left')
    merged_df['PIRACY_COUNT'] = merged_df['PIRACY_COUNT'].fillna(0)
    return merged_df


# # ⚙️ Feature Engineering
def compute_features(df, scaler):
    df['Econ_Score'] = (
        scaler.fit_transform(df[['GDP_per_capita']])[:, 0] +
        scaler.fit_transform(df[['GR']])[:, 0] +
        scaler.fit_transform(df[['INDUSTRYGDP']])[:, 0]
    ) / 3

    df['Security_Risk'] = (
        scaler.fit_transform(df[['PIRACY_COUNT']])[:, 0] +
        scaler.fit_transform(df[['CORRUPTIONINDEX']])[:, 0]
    ) / 2

    df['Fish_Score'] = scaler.fit_transform(df[['Fish_per_capita']])[:, 0]
    return df



# # 🧠 Clustering
def perform_clustering(df):
    features = df[['Econ_Score', 'Security_Risk']]
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=0)
    df['Cluster'] = kmeans.fit_predict(features_imputed)

    # Evaluate clustering
    silhouette = silhouette_score(features_imputed, df['Cluster'])
    print(f"📈 Silhouette Score (KMeans): {silhouette:.3f}")

    db = DBSCAN(eps=0.3, min_samples=10).fit(features_imputed)
    df['DBSCAN_Outlier'] = db.labels_
    return df

# # 📉 PCA for Visualization
def apply_pca(df):
    features = df[['Econ_Score', 'Security_Risk']]
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    pca = PCA(n_components=2)
    components = pca.fit_transform(features_imputed)
    df['PCA1'], df['PCA2'] = components[:, 0], components[:, 1]
    return df


# # 🧮 Multi-Criteria Decision Analysis (MCDA)
def score_countries(df):
    df['Final_Score'] = (
        0.4 * df['Econ_Score'] +
        0.3 * (1 - df['Security_Risk']) +
        0.3 * df['Fish_Score']
    )
    return df


# 🧪 Supervised Learning (Piracy Prediction ON SAME DATA)
def predict_piracy(df):
    # Drop rows with missing predictor values
    df = df.dropna(subset=['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION'])

    # Binary target: High piracy = 1 if above median
    df['High_Piracy'] = (df['PIRACY_COUNT'] > df['PIRACY_COUNT'].median()).astype(int)

    features = df[['GR', 'MILITARY', 'CORRUPTIONINDEX', 'FISHPRODUCTION']]
    target = df['High_Piracy']

    # Impute just in case
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(features)

    # Check if model already exists
    model_file = 'piracy_risk_model.joblib'
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print(f"✅ Loaded existing model from {model_file}")
    else:
        # Train new model and save
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_imputed, target)
        joblib.dump(model, model_file)
        print(f"✅ Trained and saved new model to {model_file}")

    # Predict on SAME data
    df['Predicted_High_Piracy'] = model.predict(X_imputed)
    df['Good_Candidate'] = df['Predicted_High_Piracy'] == 0

    # Show top recommended countries that are predicted low piracy risk
    top_candidates = (
        df[df['Good_Candidate']]
        .groupby('COUNTRY')['Final_Score']
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    print("\n🏆 Top Recommended Countries (Predicted Low Piracy Risk):\n", top_candidates)

    return df


# # 📊 Plotting
def plot_clusters(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
    plt.title('Country Clusters: Economic & Security Profile')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.show()

# # 🏁 Main Execution Pipeline
def main():
    economy_df, piracy_df = load_data()
    economy_df, scaler = preprocess_economy(economy_df)
    piracy_counts = aggregate_piracy(piracy_df)
    merged_df = merge_data(economy_df, piracy_counts)
    merged_df = compute_features(merged_df, scaler)
    merged_df = perform_clustering(merged_df)
    merged_df = apply_pca(merged_df)
    merged_df = score_countries(merged_df)

    # Ranking
    top_countries = merged_df.groupby('COUNTRY')['Final_Score'].mean().sort_values(ascending=False).head(15)
    print("🏆 Top Recommended Countries for Expansion:\n", top_countries)

    plot_clusters(merged_df)

    # predict_piracy(merged_df)

if __name__ == "__main__":
    main()
