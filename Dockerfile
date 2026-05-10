FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation de uv
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN chmod +x /tmp/uv-install.sh && /tmp/uv-install.sh && rm /tmp/uv-install.sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copie des fichiers de dépendances
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --system

# Copie du code source et des modèles
COPY main.py .
COPY src/ src/
COPY data/model.keras data/label.txt data/

# Lancement
CMD ["uv", "run", "python", "main.py"]