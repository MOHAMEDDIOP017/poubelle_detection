import streamlit as st
from ultralytics import YOLO
import torch
import numpy as np

from PIL import Image

# ---------------------------------------
#       CHARGEMENT DU MODELE YOLO
# ---------------------------------------
MODEL_PATH = "model/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)   # ✔ Correct pour YOLO

model = load_model()

# ---------------------------------------
#     FONCTION DE PREDICTION FULL/EMPTY
# ---------------------------------------
def predict(image):
    results = model(image)  # YOLO détecte les objets
    boxes = results[0].boxes  # bounding boxes
    
    if len(boxes) == 0:
        return None, None, None  # aucune poubelle trouvée

    # YOLO retourne les labels et scores
    first_box = boxes[0]
    cls = int(first_box.cls)
    conf = float(first_box.conf)

    # À condition que YOLO utilise 2 classes : 0=empty, 1=full
    label = model.names[cls]

    return label, conf, results[0].plot()  # image annotée


# ---------------------------------------
#       INTERFACE STREAMLIT
# ---------------------------------------
st.title("🗑️ Détection de Poubelle ")
st.write("Uploader une image pour détecter si la poubelle est **empty** ou **full**.")

uploaded_file = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image uploadée", use_container_width=True)

    if st.button("Analyser"):
        with st.spinner("Analyse en cours..."):
            label, conf, result_img = predict(img)

        if label is None:
            st.error("Aucune poubelle détectée sur l'image.")
        else:
            st.success(f"Résultat : **{label}** — Confiance : {conf:.2f}")
            st.image(result_img, caption="Détection YOLO", use_container_width=True)

# ---------------------------------------
#     TELECHARGEMENT DU MODELE
# ---------------------------------------
with open(MODEL_PATH, "rb") as f:
    st.download_button(
        label="📥 Télécharger votre modèle YOLO (.pt)",
        data=f,
        file_name="best.pt",
        mime="application/octet-stream"
    )
