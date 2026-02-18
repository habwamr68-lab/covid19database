import json
import os
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Covid High Risk Dashboard", layout="wide")

st.title("📊 Covid-19 High Risk — Dashboard")
st.write("Interface pour afficher les résultats du modèle (métriques + matrice de confusion).")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✅ Métriques d’évaluation")
    if os.path.exists("metrics.csv"):
        metrics_df = pd.read_csv("metrics.csv")
        st.dataframe(metrics_df, use_container_width=True)
    elif os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("Aucun fichier metrics.csv ou metrics.json trouvé dans le repo.")

with col2:
    st.subheader("🧩 Matrice de confusion")
    if os.path.exists("confusion_matrix.png"):
        img = Image.open("confusion_matrix.png")
        st.image(img, use_container_width=True)
    else:
        st.warning("Aucune image confusion_matrix.png trouvée dans le repo.")

st.markdown("---")
st.subheader("ℹ️ Interprétation rapide")
st.write("- **Recall** est crucial en médical (minimiser les faux négatifs).")
st.write("- Les résultats affichés correspondent au **test set (20%)**.")
