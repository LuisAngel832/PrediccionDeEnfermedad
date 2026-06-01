"""
Entrenamiento de la Red Neuronal Profunda (MLP) - Prediccion de enfermedad cardiaca
====================================================================================

Este script:
  1. Carga el dataset UCI Heart Disease (lo descarga si no existe).
  2. Separa datos de entrenamiento y prueba.
  3. Escala las variables (StandardScaler).
  4. Construye una red neuronal de VARIAS CAPAS (Deep Learning).
  5. La entrena y la evalua.
  6. Guarda el modelo, el escalador y las metricas en backend/model/.

Ejecucion (dentro de Docker o Colab, donde TensorFlow esta disponible):
    python train.py
"""

import json
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------------------------
# Rutas (funcionan tanto en local como dentro del contenedor de entrenamiento)
# ----------------------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
RUTA_DATA = os.path.join(RAIZ, "data", "heart.csv")
DIR_MODELO = os.path.join(RAIZ, "backend", "model")
URL_DATASET = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv"
)

# Nombres y orden EXACTO de las 13 variables de entrada (el backend usa el mismo orden)
COLUMNAS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def cargar_datos():
    """Carga el CSV; si no existe lo descarga del repositorio publico."""
    os.makedirs(os.path.dirname(RUTA_DATA), exist_ok=True)
    if not os.path.exists(RUTA_DATA):
        print(f"[INFO] Dataset no encontrado. Descargando desde:\n  {URL_DATASET}")
        urllib.request.urlretrieve(URL_DATASET, RUTA_DATA)
        print("[INFO] Descarga completa.")
    df = pd.read_csv(RUTA_DATA)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def construir_modelo(n_entradas: int) -> tf.keras.Model:
    """Red neuronal profunda: VARIAS capas densas ocultas + Dropout (regularizacion)."""
    modelo = models.Sequential(
        [
            layers.Input(shape=(n_entradas,)),
            layers.Dense(64, activation="relu"),   # Capa oculta 1
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),   # Capa oculta 2
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),   # Capa oculta 3
            layers.Dense(1, activation="sigmoid"), # Salida: probabilidad (0 a 1)
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    df = cargar_datos()

    X = df[COLUMNAS].values

    # IMPORTANTE: en este dataset (heart.csv de UCI/Kaggle) la columna 'target'
    # esta INVERTIDA respecto a la intuicion: target=1 corresponde a pacientes
    # SANOS y target=0 a pacientes con enfermedad (se confirma al ver que la clase
    # target=1 tiene mejor frecuencia cardiaca, menos angina y menos vasos afectados).
    # Para que nuestra etiqueta '1' signifique "tiene enfermedad" (coherente con el
    # frontend y la interpretacion clinica), invertimos: enfermedad = 1 - target.
    y = 1 - df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado: las redes neuronales aprenden mejor con datos normalizados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = construir_modelo(X_train.shape[1])
    modelo.summary()

    # EarlyStopping: detiene el entrenamiento si deja de mejorar (evita sobreajuste)
    early = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    print("\n[INFO] Entrenando la red neuronal...\n")
    historia = modelo.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[early],
        verbose=2,
    )

    # ------------------------------------------------------------------
    # Evaluacion
    # ------------------------------------------------------------------
    y_prob = modelo.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    matriz = confusion_matrix(y_test, y_pred).tolist()
    reporte = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n[RESULTADO] Accuracy en prueba: {acc:.4f}")
    print("[RESULTADO] Matriz de confusion:")
    print(np.array(matriz))
    print("\n[RESULTADO] Reporte de clasificacion:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # Guardado de artefactos en backend/model/
    # ------------------------------------------------------------------
    os.makedirs(DIR_MODELO, exist_ok=True)
    modelo.save(os.path.join(DIR_MODELO, "modelo_corazon.keras"))
    joblib.dump(scaler, os.path.join(DIR_MODELO, "scaler.pkl"))

    metricas = {
        "accuracy": round(acc, 4),
        "n_filas": int(df.shape[0]),
        "n_variables": len(COLUMNAS),
        "columnas": COLUMNAS,
        "matriz_confusion": matriz,
        "precision_clase_1": round(reporte["1"]["precision"], 4),
        "recall_clase_1": round(reporte["1"]["recall"], 4),
        "epocas_entrenadas": len(historia.history["loss"]),
        "arquitectura": "MLP profunda: 64 -> 32 -> 16 -> 1 (sigmoid)",
    }
    with open(os.path.join(DIR_MODELO, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Modelo y artefactos guardados en: {DIR_MODELO}")
    print("     - modelo_corazon.keras")
    print("     - scaler.pkl")
    print("     - metrics.json")


if __name__ == "__main__":
    main()
