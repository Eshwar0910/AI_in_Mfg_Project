# Use a specific, stable Python base. Change to python:3.12-slim if you know TF supports it in your environment.
FROM python:3.10-slim

# Install system deps required by some packages (OpenCV, etc.) and for efficient pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies first (cache layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default command — run Streamlit on 0.0.0.0 so it's accessible from the container
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
