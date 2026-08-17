FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH="/app/core:/app/core/model:${PYTHONPATH}"

# System deps + PETSc
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenmpi-dev \
    petsc-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python deps FIRST (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# THEN copy app code (changes often, but pip layer stays cached)
COPY app/ app/
COPY core/ core/
COPY models.json .
COPY test_api.py .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]