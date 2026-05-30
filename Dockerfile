# Use the official Python image as base
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema (para PyTorch, OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt /app/

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY src ./src
COPY models ./models

# Exponer puerto
EXPOSE 8080

# Comando para correr FastAPI correctamente
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]