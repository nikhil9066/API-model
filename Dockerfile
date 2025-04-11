# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (if needed, for packages like GDAL, libzmq, etc.)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment in the container
RUN python -m venv /env
ENV PATH="/env/bin:$PATH"

# Install any needed dependencies specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable for Python to ensure it's using the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Expose the port the app runs on (replace with your app's port)
EXPOSE 5000

# Run your application (replace this with the command to run your app)
CMD ["python", "app.py"]  # Replace with the command you use to run the app