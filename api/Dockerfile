FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ /app/api/
COPY model/ /app/model/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "api/app.py"]

