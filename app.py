from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Initialisation de l'application FastAPI
app = FastAPI()

# Chemin vers le modèle sauvegardé
MODEL_PATH = "model.pkl"

# Chargement du modèle avec pickle
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Modèle chargé avec succès.")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")

# Définition du schéma d'entrée via Pydantic
class PredictionInput(BaseModel):
    features: list

# Route GET pour tester l'API
@app.get("/")
def read_root():
    return {"message": "API de prédiction est en ligne !"}

# Route POST pour effectuer une prédiction
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Vérification des données entrantes
        if not input_data.features:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie pour la prédiction.")

        # Conversion en format numpy
        data = np.array(input_data.features).reshape(1, -1)

        # Prédiction avec le modèle
        prediction = model.predict(data)

        # Retourner le résultat
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

