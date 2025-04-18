# Use the official Python image as the base image
FROM python:3.10-slim

# Install the required system packages
RUN apt-get update && \
    apt-get install -y gcc libffi-dev build-essential && \
    apt-get clean

# Set the working directory to /app
WORKDIR /app

# Set the build argument for the app version number
ARG APP_VERSION=0.1.0

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5001 for the FastAPI application
EXPOSE 5001

# Set the environment variable for the app version number
ENV APP_VERSION=$APP_VERSION

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
