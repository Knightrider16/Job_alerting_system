import streamlit as st
import pandas as pd
import joblib
from train_and_cluster import scrape_karkidi_jobs, generate_cluster_names

# Load vectorizer and model once
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("kmeans_model.pkl")

st.title("Karkidi Job Alert System")

if st.button("Update Jobs"):
    with st.spinner("Looking for new jobs......"):
        df = scrape_karkidi_jobs(keyword="data science", pages=1)
        df = df[df['Skills'].str.strip() != ""]
        X = vectorizer.transform(df['Skills'])
        df['Cluster'] = model.predict(X)
        cluster_name_map = generate_cluster_names(df)
        df['Cluster_Name'] = df['Cluster'].map(cluster_name_map)
        df.to_csv("clustered_jobs.csv", index=False)
        st.success(f"There are a total of {len(df)} jobs that are of interest for you.")
        st.dataframe(df[['Title', 'Company', 'Location', 'Cluster_Name']])

st.header("Browse Jobs by Interest")

try:
    df_existing = pd.read_csv("clustered_jobs.csv")
    cluster_options = sorted(df_existing['Cluster_Name'].unique())
    selected_cluster = st.selectbox("Select Domain", cluster_options)
    filtered = df_existing[df_existing['Cluster_Name'] == selected_cluster]
    st.dataframe(filtered[['Title', 'Company', 'Location', 'Skills']])
except FileNotFoundError:
    st.warning("No clustered job data found. Please update jobs first.")
