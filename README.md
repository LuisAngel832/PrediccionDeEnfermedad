# ❤️ Proyecto IA #4 — Deep Learning (varias capas): Predicción de enfermedad cardíaca

Aplicación web completa (Frontend + Backend + entrenamiento) que usa una **red neuronal
profunda (MLP con varias capas ocultas)** para estimar el riesgo de enfermedad cardíaca
a partir de 13 variables clínicas.

> Proyecto académico. **No usa ninguna API de IA de pago**: el modelo se entrena localmente
> (en Docker o Google Colab) con librerías gratuitas.

---

## 📚 ¿Qué es esto en palabras simples?

- **Deep Learning** = redes neuronales con **varias capas**. Cada capa aprende patrones cada
  vez más abstractos a partir de los datos.
- Aquí la red recibe 13 datos de un paciente (edad, presión, colesterol, etc.) y devuelve una
  **probabilidad de 0 a 100 %** de tener enfermedad cardíaca.
- La arquitectura es: `13 entradas → 64 → 32 → 16 neuronas → 1 salida (probabilidad)`.

---

## 🗂️ Estructura del repositorio (monorepo)

```
proyectoIa/
├── data/heart.csv          # Dataset de entrenamiento (UCI Heart Disease)
├── training/               # Entrenamiento de la red neuronal
│   ├── train.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── Colab_Entrenamiento.ipynb
├── backend/                # API que sirve el modelo (FastAPI)
│   ├── app/main.py
│   ├── model/              # Modelo entrenado (se genera en el entrenamiento)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # Interfaz web (HTML + CSS + JS, servida con Nginx)
│   ├── index.html, style.css, app.js
│   ├── nginx.conf
│   └── Dockerfile
├── docs/MEMORIA.md         # Plantilla de la Memoria PDF
└── docker-compose.yml
```

---

## 🚀 Cómo ejecutarlo (todo con Docker, sin instalar Python)

### Paso 1 — Entrenar el modelo
Genera `backend/model/modelo_corazon.keras`, `scaler.pkl` y `metrics.json`.

```bash
docker compose run --rm trainer
```

> Alternativa sin Docker: abre `training/Colab_Entrenamiento.ipynb` en
> [Google Colab](https://colab.research.google.com/), ejecútalo y descarga los 3 archivos
> generados a la carpeta `backend/model/`.

### Paso 2 — Levantar la aplicación
```bash
docker compose up --build
```

- **Frontend:** http://localhost:8080
- **Backend (API):** http://localhost:8000 — documentación interactiva en http://localhost:8000/docs

### Paso 3 — Probar
Abre http://localhost:8080, llena el formulario y pulsa **Predecir**.

Para detener: `Ctrl + C` y luego `docker compose down`.

---

## 🧪 Probar la API directamente

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":63,\"sex\":1,\"cp\":3,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":0,\"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":0,\"ca\":0,\"thal\":1}"
```

Respuesta de ejemplo:
```json
{ "probabilidad": 0.87, "porcentaje": 87.0, "tiene_riesgo": true,
  "resultado": "Riesgo de enfermedad cardiaca" }
```

---

## 🐳 Publicar imágenes en DockerHub (entregable)

Reemplaza `TU_USUARIO` por tu usuario de DockerHub.

```bash
# 1) Iniciar sesión
docker login

# 2) Etiquetar las imágenes ya construidas
docker tag heart-backend:1.0  TU_USUARIO/heart-backend:1.0
docker tag heart-frontend:1.0 TU_USUARIO/heart-frontend:1.0

# 3) Subirlas
docker push TU_USUARIO/heart-backend:1.0
docker push TU_USUARIO/heart-frontend:1.0
```

Luego cualquiera puede usarlas sin construir nada:
```bash
docker run -d -p 8000:8000 TU_USUARIO/heart-backend:1.0
docker run -d -p 8080:80   TU_USUARIO/heart-frontend:1.0
```

---

## 🧠 Dataset

[UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (~303 registros, 13
variables + objetivo `target`). Incluido en `data/heart.csv`.

| Variable | Significado |
|---|---|
| age | Edad |
| sex | Sexo (1 hombre, 0 mujer) |
| cp | Tipo de dolor de pecho (0-3) |
| trestbps | Presión arterial en reposo |
| chol | Colesterol (mg/dl) |
| fbs | Azúcar en ayunas > 120 (1/0) |
| restecg | Electrocardiograma en reposo (0-2) |
| thalach | Frecuencia cardíaca máxima |
| exang | Angina por ejercicio (1/0) |
| oldpeak | Depresión del ST |
| slope | Pendiente del ST (0-2) |
| ca | Vasos principales (0-4) |
| thal | Talasemia (0-3) |
| **target** | Objetivo original del CSV (⚠️ en este dataset `1 = sano`, `0 = enfermo`). El entrenamiento lo **invierte** a `enfermedad = 1 - target` para que `1` signifique enfermedad. |

---

## ⚖️ Aviso

Proyecto educativo. **No** es una herramienta de diagnóstico médico real.
