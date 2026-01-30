FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build frontend if needed
RUN if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    cd frontend && npm install && npm run build; \
    fi

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run_app.py"]
