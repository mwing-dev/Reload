FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/mwing-dev/Reload_Trained_Model.git /app/Reload_Trained_Model

ENV REPO_PATH=/app/Reload_Trained_Model

COPY app.py .

RUN pip install gradio pyserial

EXPOSE 7860

CMD ["python", "app.py"]
