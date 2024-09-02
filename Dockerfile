# Etapa 1: Construcción de dependencias
FROM python:3.11-slim AS build-stage

WORKDIR /app

# Copia y instala las dependencias de la aplicación
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Etapa 2: Imagen final
FROM python:3.11-slim

WORKDIR /app

# Copia solo las dependencias instaladas desde la etapa anterior
COPY --from=build-stage /install /usr/local

# Copia el resto de la aplicación al contenedor
COPY . .

EXPOSE 5001

CMD ["python", "main.py"]
