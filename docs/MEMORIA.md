# Memoria del Proyecto — IA #4

> **Plantilla.** Rellena los campos entre `[corchetes]`, pega tus capturas donde se indica y
> exporta a **PDF** (en VS Code: extensión *Markdown PDF*, o copia a Word/Google Docs).

---

## Portada

- **Materia:** [Nombre de la materia]
- **Proyecto:** #4 — Deep Learning (red neuronal de varias capas)
- **Título:** Predicción de enfermedad cardíaca con una red neuronal profunda
- **Integrantes:** [Nombres]
- **Profesor:** [Nombre]
- **Fecha:** [Fecha de entrega]

---

## 2.1 Descripción del proyecto (1 a 2 cuartillas)

Este proyecto implementa una aplicación web que predice el **riesgo de enfermedad cardíaca**
de un paciente a partir de 13 variables clínicas (edad, sexo, presión arterial, colesterol,
frecuencia cardíaca máxima, etc.). El núcleo es una **red neuronal profunda (Deep Learning)**
del tipo Perceptrón Multicapa (MLP), compuesta por **varias capas densas ocultas**.

**¿Por qué Deep Learning de varias capas?** A diferencia de un modelo de Machine Learning
clásico (una sola etapa de decisión), una red profunda apila varias capas de neuronas. Cada
capa transforma la información y aprende representaciones cada vez más abstractas, lo que le
permite capturar relaciones no lineales entre las variables clínicas.

**Arquitectura de la red:**

```
Entrada (13 variables)
  → Capa Densa 64 neuronas (ReLU) → Dropout 30 %
  → Capa Densa 32 neuronas (ReLU) → Dropout 20 %
  → Capa Densa 16 neuronas (ReLU)
  → Capa de salida 1 neurona (Sigmoid) → probabilidad de 0 a 1
```

**Tecnologías utilizadas:**
- **Entrenamiento:** Python, TensorFlow/Keras, scikit-learn, pandas.
- **Backend:** FastAPI (API REST) + Uvicorn.
- **Frontend:** HTML, CSS y JavaScript, servidos con Nginx.
- **Contenerización:** Docker y Docker Compose.
- **Dataset:** UCI Heart Disease (~303 registros).

**Objetivo:** demostrar el ciclo completo de un proyecto de Deep Learning: datos → entrenamiento
→ despliegue de un servicio (backend) → interfaz de usuario (frontend), todo contenerizado.

---

## 2.2 Proceso de implementación y despliegue

**1. Datos.** Se utilizó el dataset público UCI Heart Disease (`data/heart.csv`).

**2. Entrenamiento (`training/train.py`).**
- División 80 % entrenamiento / 20 % prueba.
- Normalización de variables con `StandardScaler`.
- Construcción y entrenamiento de la red (200 épocas máx., con *EarlyStopping*).
- Se guardan: el modelo (`modelo_corazon.keras`), el escalador (`scaler.pkl`) y las
  métricas (`metrics.json`).

> 📷 **Captura:** salida de la consola durante el entrenamiento (epochs y accuracy).
>
> `[ Pega aquí la captura ]`

**3. Backend (`backend/app/main.py`).** API con FastAPI que carga el modelo y expone:
`GET /health`, `GET /info`, `POST /predict`.

**4. Frontend (`frontend/`).** Formulario web que envía los datos al backend vía `/api` (proxy
de Nginx) y muestra la probabilidad con una barra de color.

**5. Contenerización y despliegue.**
- Cada componente tiene su `Dockerfile`.
- `docker-compose.yml` orquesta todo.
- Comandos:
  ```bash
  docker compose run --rm trainer   # entrenar
  docker compose up --build         # levantar la app
  ```

> 📷 **Captura:** terminal con `docker compose up` y los contenedores corriendo.
>
> `[ Pega aquí la captura ]`

---

## 2.3 Funcionalidad (capturas de pantalla)

> 📷 **Captura 1:** la interfaz web (http://localhost:8080) con el formulario.
>
> `[ Pega aquí la captura ]`

> 📷 **Captura 2:** resultado de una predicción de **riesgo** (barra roja).
>
> `[ Pega aquí la captura ]`

> 📷 **Captura 3:** resultado de una predicción **sin riesgo** (barra verde).
>
> `[ Pega aquí la captura ]`

> 📷 **Captura 4:** documentación interactiva del backend (http://localhost:8000/docs).
>
> `[ Pega aquí la captura ]`

---

## 2.4 Rutas de código (GitHub) e imágenes Docker

**Repositorio GitHub (monorepo):**
- URL: https://github.com/LuisAngel832/PrediccionDeEnfermedad
- Entrenamiento: `training/train.py`
- Backend: `backend/app/main.py`
- Frontend: `frontend/index.html`, `frontend/app.js`
- Orquestación: `docker-compose.yml`

**Imágenes en DockerHub:**
- Backend: https://hub.docker.com/r/luisangel832/heart-backend → `luisangel832/heart-backend:1.0`
- Frontend: https://hub.docker.com/r/luisangel832/heart-frontend → `luisangel832/heart-frontend:1.0`

> 📷 **Captura:** las imágenes publicadas en tu cuenta de DockerHub.
>
> `[ Pega aquí la captura ]`

---

## Resultados del modelo

- **Accuracy en prueba:** [ver `backend/model/metrics.json`]
- **Matriz de confusión:** [ver `metrics.json`]
- **Arquitectura:** MLP 64 → 32 → 16 → 1 (sigmoid)

---

## Conclusiones

[Escribe 1-2 párrafos: qué aprendiste, qué dificultades tuviste, cómo podría mejorarse el
modelo (más datos, más capas, ajuste de hiperparámetros, etc.).]
