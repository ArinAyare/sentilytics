# Build from repo root:  docker build -f backend/Dockerfile -t sentilytics-api .
FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pymongo requests pandas scikit-learn joblib
COPY backend/app ./app
COPY ml/__init__.py ml/common.py ./ml/
COPY artifacts ./artifacts
ENV PYTHONPATH=/srv ARTIFACTS_DIR=/srv/artifacts LOCAL_MODEL_PATH=/srv/artifacts/baseline/model.joblib
RUN useradd -m app && chown -R app /srv
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
