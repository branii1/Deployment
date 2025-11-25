FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY netflix_type_rf_model.pkl .
COPY predict.py .

EXPOSE 8080

CMD ["python", "predict.py"]
