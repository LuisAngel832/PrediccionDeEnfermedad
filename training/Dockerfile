# Imagen para ENTRENAR la red neuronal (no necesita tu Python local).
# Usa Python 3.11 porque TensorFlow aun no soporta Python 3.14.
FROM python:3.11-slim

WORKDIR /app

# Dependencias de entrenamiento
COPY training/requirements.txt training/requirements.txt
RUN pip install --no-cache-dir -r training/requirements.txt

# Codigo de entrenamiento (la estructura imita la del repo: train.py usa rutas
# relativas a /app -> /app/data y /app/backend/model).
COPY training/train.py training/train.py

# El dataset y la salida se montan como volumenes desde docker-compose:
#   ./data          -> /app/data            (entrada; si falta, train.py lo descarga)
#   ./backend/model -> /app/backend/model   (salida: modelo, scaler, metrics)
CMD ["python", "training/train.py"]
