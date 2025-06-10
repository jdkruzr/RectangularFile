FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for pdf2image
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /mnt/rectangularfile/documents /mnt/rectangularfile/debug_images /mnt/rectangularfile/qwencache

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run with Gunicorn
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]