# =============================================================================
# Stage 1: Build Frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

# Copy frontend files
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Install Python Dependencies
# =============================================================================
FROM python:3.11-slim AS python-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-railway.txt .
RUN pip install --no-cache-dir --user -r requirements-railway.txt

# =============================================================================
# Stage 3: Final Production Image
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=python-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy application code
COPY . .

# Copy built frontend assets from frontend-builder
# Note: Vite builds to ../static (relative to frontend), so in Docker it's at /static
COPY --from=frontend-builder /static/assets ./static/assets
COPY --from=frontend-builder /static/index.html ./static/index.html
COPY --from=frontend-builder /static/favicon.svg ./static/favicon.svg

# Expose port (Railway sets PORT env var)
EXPOSE 8080

# Run the application
CMD ["python", "run_app.py"]
