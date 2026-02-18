import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# CONFIGURATION PAGE
# ===============================
st.set_page_config(
    page_title="Prédiction Risque Covid-19",
    page_icon="🩺",
    layout="centered"
)

# ===============================
# STYLE (COULEURS VIVES)
# ===============================
st.markdown("""
    <style>
    body {
        background-color: #f4f9ff;
    }
    h1 {
        color: #0d47a1;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff5722;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# TITRE
# ===============================
st.title("🩺 Prédiction du Risque Covid-19")
st.write("Entrez les informations du patient pour estimer son niveau de risque.")

# ===============================
# CHARGEMENT MODELE
# ===============================
try:
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
except:
    st.error("Erreur : model.pkl ou features.pkl introuvable.")
    st.stop()

# ===============================
# FORMULAIRE UTILISATEUR
# ===============================

age = st.slider("Âge", 0, 100, 40)

sexe = st.selectbox("Sexe", ["Femme", "Homme"])
sexe = 0 if sexe == "Femme" else 1

diabete = st.selectbox("Diabète", ["Non", "Oui"])
diabete = 1 if diabete == "Oui" else 0

hypertension = st.selectbox("Hypertension", ["Non", "Oui"])
hypertension = 1 if hypertension == "Oui" else 0

obesite = st.selectbox("Obésité", ["Non", "Oui"])
obesite = 1 if obesite == "Oui" else 0

asthme = st.selectbox("Asthme", ["Non", "Oui"])
asthme = 1 if asthme == "Oui" else 0

pneumonie = st.selectbox("Pneumonie", ["Non", "Oui"])
pneumonie = 1 if pneumonie == "Oui" else 0

# ===============================
# BOUTON PREDICTION
# ===============================

if st.button("🔍 Prédire le risque"):

    # Créer dictionnaire entrée
    input_dict = {
        "age": age,
        "sex": sexe,
        "diabetes": diabete,
        "hypertension": hypertension,
        "obesity": obesite,
        "asthma": asthme,
        "pneumonia": pneumonie
    }

    # Convertir en DataFrame
    input_df = pd.DataFrame([input_dict])

    # Appliquer get_dummies comme au training
    input_df = pd.get_dummies(input_df, dummy_na=True)

    # Ajouter colonnes manquantes
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Réordonner colonnes EXACTEMENT comme au training
    input_df = input_df[features]

    # Prédiction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ RISQUE ÉLEVÉ détecté")
        st.write(f"Probabilité estimée : **{probability:.2%}**")
    else:
        st.success("✅ RISQUE FAIBLE détecté")
        st.write(f"Probabilité estimée : **{probability:.2%}**")

    st.markdown("---")
    st.info("⚠️ Ceci est une estimation basée sur un modèle statistique et ne remplace pas un avis médical.")
