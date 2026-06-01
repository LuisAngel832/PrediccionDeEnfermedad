# Imagen del BACKEND (API FastAPI que sirve el modelo entrenado).
FROM python:3.11-slim

WORKDIR /app

# Dependencias
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la API y el modelo entrenado
COPY backend/app app
COPY backend/model model

EXPOSE 8000

# Arranca el servidor. 0.0.0.0 para que sea accesible desde fuera del contenedor.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
