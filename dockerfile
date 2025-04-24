FROM python:3.10-slim

WORKDIR /app

RUN mkdir -p /app/uploads
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
