FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
COPY train_model.py .
RUN pip install --no-cache-dir -r requirements.txt
RUN python train_model.py
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]