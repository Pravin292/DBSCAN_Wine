import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Wine Cluster Identifier",
    page_icon="🍷",
    layout="centered"
)

# ---------------- Styling ----------------
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: white;
}
.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #800000;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
}
div.stButton > button {
    background-color: #800000;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
div.stButton > button:hover {
    background-color: #a00000;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🍷 Wine Cluster Identifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Density-Based Classification using DBSCAN</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- Load Data & Model Setup ----------------
df = pd.read_csv("wine_clustering_data.csv")
scaler = joblib.load("wine_scaler.pkl")

# Scale full dataset
X_scaled = scaler.transform(df)

# Refit DBSCAN with same parameters used during training
eps = 2
min_samples = 2

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(X_scaled)

# Store cluster centers (mean of each cluster)
clusters = {}
for label in set(cluster_labels):
    if label != -1:
        clusters[label] = X_scaled[cluster_labels == label]

# ---------------- Input Fields ----------------
col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol", 0.0, 20.0, 13.0)
    malic_acid = st.number_input("Malic Acid", 0.0, 10.0, 2.0)
    ash = st.number_input("Ash", 0.0, 5.0, 2.5)
    ash_alcanity = st.number_input("Ash Alcanity", 0.0, 30.0, 15.0)
    magnesium = st.number_input("Magnesium", 0.0, 200.0, 100.0)
    total_phenols = st.number_input("Total Phenols", 0.0, 5.0, 2.5)
    flavanoids = st.number_input("Flavanoids", 0.0, 5.0, 2.0)

with col2:
    nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", 0.0, 5.0, 0.3)
    proanthocyanins = st.number_input("Proanthocyanins", 0.0, 5.0, 1.5)
    color_intensity = st.number_input("Color Intensity", 0.0, 15.0, 5.0)
    hue = st.number_input("Hue", 0.0, 2.0, 1.0)
    od280 = st.number_input("OD280", 0.0, 5.0, 3.0)
    proline = st.number_input("Proline", 0.0, 2000.0, 1000.0)

# ---------------- Prediction Logic ----------------
if st.button("🔍 Identify Cluster"):

    input_data = np.array([[alcohol, malic_acid, ash, ash_alcanity,
                            magnesium, total_phenols, flavanoids,
                            nonflavanoid_phenols, proanthocyanins,
                            color_intensity, hue, od280, proline]])

    scaled_input = scaler.transform(input_data)

    assigned_cluster = -1
    min_distance = float("inf")

    # Check distance to each cluster
    for label, cluster_points in clusters.items():
        distances = euclidean_distances(scaled_input, cluster_points)
        closest_distance = np.min(distances)

        if closest_distance < eps and closest_distance < min_distance:
            min_distance = closest_distance
            assigned_cluster = label

    st.markdown("---")

    if assigned_cluster == -1:
        st.error("⚠️ This wine is classified as Noise / Outlier")
    else:
        st.success(f"🍷 This wine belongs to Cluster {assigned_cluster}")
        st.write(f"Distance to cluster core: {min_distance:.3f}")