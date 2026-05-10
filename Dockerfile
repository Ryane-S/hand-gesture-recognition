FROM python:3.11-slim

# Éviter les dialogues interactifs pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système (nécessite root)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation de uv (toujours en root)
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN chmod +x /tmp/uv-install.sh && /tmp/uv-install.sh && rm /tmp/uv-install.sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copier uniquement les fichiers de dépendances (root)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copier le code source et les modèles (root)
COPY main.py .
COPY src/ src/
COPY data/model.keras data/label.txt data/

# Créer un utilisateur non‑root (avec un homedir)
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

# Passer à l'utilisateur non‑root pour l'exécution
USER appuser

# Lancer l'application
CMD ["uv", "run", "python", "main.py"]