# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /pepNmemb

COPY . /pepNmemb/

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Install the package in editable mode
RUN pip install -e .

# Optional: Run tests
RUN pip install pytest
RUN pytest tests/

# Default command (can be overridden)
CMD ["python", "-m", "pepNmemb"]