FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (so Docker caches the install layer)
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Start the app with gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
