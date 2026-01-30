# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements-railway.txt .
RUN pip install --no-cache-dir --user -r requirements-railway.txt

# Final stage - smaller image
FROM python:3.11-slim

WORKDIR /app

# Copy only runtime dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run_app.py"]
