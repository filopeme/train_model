FROM python:3.9

WORKDIR /app

COPY . /app

# Fix permissions for upload folder
# RUN chmod -R 755 /app/uploads

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch (CPU version)
RUN pip install torch==2.0.1 torchvision==0.15.2

# Install remaining dependencies
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

