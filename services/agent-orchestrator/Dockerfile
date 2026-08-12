FROM python:3.11-slim AS base
WORKDIR /app
COPY pyproject.toml ./
COPY app.py config.py ollama_client.py ./
COPY graphs ./graphs
COPY prompts ./prompts
RUN pip install --no-cache-dir .
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]