# Dockerfile for Render.com deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY app ./app

# Create directory for model weights (if needed)
RUN mkdir -p app/models/weights

# Optional: Pre-download model weights
# RUN wget -O app/models/weights/xception.pth https://your-link.com/xception.pth

EXPOSE 8000

# Health check (optional)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
