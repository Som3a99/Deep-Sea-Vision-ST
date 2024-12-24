FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
COPY packages.txt .
RUN apt-get update && \
    apt-get install -y $(cat packages.txt) && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directory for weights
RUN mkdir -p weights/detection

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]