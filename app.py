import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# CONFIG PAGE
# ===============================
st.set_page_config(
    page_title="Prédiction Risque Covid",
    page_icon="🩺",
    layout="centered"
)

# ===============================
# STYLE CSS (couleurs vives)
# ===============================
st.markdown("""
    <style>
    .main {
        background-color: #f4f9ff;
    }
    h1 {
        color: #0d47a1;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff5722;
        color: white;
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
# CHARGER MODELE
# ===============================
model = joblib.load("model.pkl")

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

    # Créer dataframe (doit correspondre EXACTEMENT aux features utilisées à l'entraînement)
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sexe,
        "diabetes": diabete,
        "hypertension": hypertension,
        "obesity": obesite,
        "asthma": asthme,
        "pneumonia": pneumonie
    }])

    # Adapter si ton modèle attend d'autres colonnes
    # (sinon erreur)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ Risque ÉLEVÉ détecté")
        st.write(f"Probabilité estimée : **{probability:.2%}**")
    else:
        st.success("✅ Risque FAIBLE détecté")
        st.write(f"Probabilité estimée : **{probability:.2%}**")

    st.markdown("---")
    st.write("⚠️ Ceci est une estimation basée sur un modèle statistique et ne remplace pas un avis médical.")
