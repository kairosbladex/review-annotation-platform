FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
COPY backend ./backend
COPY migrations ./migrations
COPY alembic.ini ./

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
