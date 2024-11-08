# Use a base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements directly (optional, to avoid reinstallation if requirements change infrequently)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone the public GitHub repository
RUN git clone https://github.com/mwing-dev/Reload_Trained_Model.git /app/Reload_Trained_Model

# Set the repository path as an environment variable
ENV REPO_PATH=/app/Reload_Trained_Model

# Copy the Python app code into the container
COPY app.py .

# Install additional Python packages
RUN pip install gradio pyserial

# Expose the port for Gradio
EXPOSE 7860

# Define the command to run the app
CMD ["python", "app.py"]
