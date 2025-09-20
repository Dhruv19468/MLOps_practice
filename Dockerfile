# Use official lightweight Python image
FROM python:3.12-slim

# Set work directory inside the container
WORKDIR /app

# Copy dependency files first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run Flask app
CMD ["python", "api/app.py"]