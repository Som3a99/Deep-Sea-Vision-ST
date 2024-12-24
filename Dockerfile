# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
COPY packages.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends $(cat packages.txt) && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p weights/detection && \
    mkdir -p .streamlit

# Set up streamlit configuration
COPY .streamlit/config.toml .streamlit/

# Expose port
EXPOSE 8501

# Set environment variables for streamlit
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Run the application
CMD ["streamlit", "run", "app.py"]