# Scoring API: docker build -t credit-risk-scoring . && docker run -p 8000:8000 credit-risk-scoring
FROM python:3.11-slim

# XGBoost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Code, configuration and the trained models (run `make train` before building)
COPY configs/ configs/
COPY src/ src/
COPY models/scorecard.joblib models/calibrators.joblib models/decision.json models/
ENV PYTHONPATH=/app/src

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "credit_risk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
