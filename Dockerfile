FROM python:3.11-bullseye

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicron", "main:app", "--host", "0.0.0.0", "--port", "8000"]
