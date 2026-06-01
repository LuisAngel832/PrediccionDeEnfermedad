"""
Backend - API de prediccion de enfermedad cardiaca
===================================================

API hecha con FastAPI que:
  - Carga el modelo de red neuronal entrenado (modelo_corazon.keras) y el escalador.
  - Expone endpoints para consultar estado, informacion y hacer predicciones.

Endpoints:
  GET  /health   -> estado del servicio
  GET  /info     -> metricas del modelo y descripcion de las variables
  POST /predict  -> recibe las 13 variables clinicas y devuelve la prediccion
"""

import json
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import tensorflow as tf

# ----------------------------------------------------------------------------
# Carga de artefactos del modelo
# ----------------------------------------------------------------------------
DIR_MODELO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
RUTA_MODELO = os.path.join(DIR_MODELO, "modelo_corazon.keras")
RUTA_SCALER = os.path.join(DIR_MODELO, "scaler.pkl")
RUTA_METRICAS = os.path.join(DIR_MODELO, "metrics.json")

modelo = None
scaler = None
metricas = {}

# Descripcion legible de cada variable (se muestra en /info y en el front)
DESCRIPCION_VARIABLES = {
    "age": "Edad en anios",
    "sex": "Sexo (1 = hombre, 0 = mujer)",
    "cp": "Tipo de dolor de pecho (0-3)",
    "trestbps": "Presion arterial en reposo (mm Hg)",
    "chol": "Colesterol serico (mg/dl)",
    "fbs": "Azucar en ayunas > 120 mg/dl (1 = si, 0 = no)",
    "restecg": "Resultado electrocardiograma en reposo (0-2)",
    "thalach": "Frecuencia cardiaca maxima alcanzada",
    "exang": "Angina inducida por ejercicio (1 = si, 0 = no)",
    "oldpeak": "Depresion del ST inducida por ejercicio",
    "slope": "Pendiente del segmento ST (0-2)",
    "ca": "Numero de vasos principales coloreados (0-4)",
    "thal": "Talasemia (0-3)",
}

ORDEN_COLUMNAS = list(DESCRIPCION_VARIABLES.keys())


def cargar_modelo():
    """Carga modelo, scaler y metricas en variables globales."""
    global modelo, scaler, metricas
    if not os.path.exists(RUTA_MODELO):
        raise RuntimeError(
            f"No se encontro el modelo en {RUTA_MODELO}. "
            "Ejecuta primero el entrenamiento (training/train.py o el servicio 'trainer')."
        )
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    scaler = joblib.load(RUTA_SCALER)
    if os.path.exists(RUTA_METRICAS):
        with open(RUTA_METRICAS, encoding="utf-8") as f:
            metricas = json.load(f)


# ----------------------------------------------------------------------------
# Esquema de entrada (validacion automatica con Pydantic)
# ----------------------------------------------------------------------------
class Paciente(BaseModel):
    age: float = Field(..., example=63)
    sex: int = Field(..., ge=0, le=1, example=1)
    cp: int = Field(..., ge=0, le=3, example=3)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: int = Field(..., ge=0, le=1, example=1)
    restecg: int = Field(..., ge=0, le=2, example=0)
    thalach: float = Field(..., example=150)
    exang: int = Field(..., ge=0, le=1, example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., ge=0, le=2, example=0)
    ca: int = Field(..., ge=0, le=4, example=0)
    thal: int = Field(..., ge=0, le=3, example=1)


# ----------------------------------------------------------------------------
# Aplicacion
# ----------------------------------------------------------------------------
app = FastAPI(
    title="API - Prediccion de enfermedad cardiaca",
    description="Red neuronal profunda (MLP) que estima el riesgo de enfermedad cardiaca.",
    version="1.0.0",
)

# CORS abierto: permite que el frontend (otro origen) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    cargar_modelo()


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": modelo is not None}


@app.get("/info")
def info():
    return {
        "modelo": "Red neuronal profunda (MLP) con varias capas",
        "metricas": metricas,
        "variables": DESCRIPCION_VARIABLES,
    }


@app.post("/predict")
def predict(paciente: Paciente):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    # Construir el vector en el ORDEN correcto de columnas
    datos = paciente.model_dump()
    vector = np.array([[datos[col] for col in ORDEN_COLUMNAS]], dtype=float)

    # Escalar con el MISMO scaler del entrenamiento
    vector_esc = scaler.transform(vector)

    probabilidad = float(modelo.predict(vector_esc).ravel()[0])
    tiene_riesgo = probabilidad >= 0.5

    return {
        "probabilidad": round(probabilidad, 4),
        "porcentaje": round(probabilidad * 100, 2),
        "tiene_riesgo": tiene_riesgo,
        "resultado": "Riesgo de enfermedad cardiaca" if tiene_riesgo
        else "Sin riesgo significativo",
    }
