# 1. The Base Blueprint (Lightweight Linux with Python 3.11)
FROM python:3.11-slim

# 2. Set up our working directory inside the Docker image
WORKDIR /app

# 3. Copy the requirements file and install Python libraries
COPY requirements.txt .
# We keep the PyTorch CPU flag so it doesn't try to download 5GB of Nvidia GPU drivers
RUN pip install -r requirements.txt --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/cpu

# 4. The Machine Learning Pre-bake
# Downloads BGE-small directly into the frozen image, saving startup time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 5. Copy all the code into the container
COPY . .

# 6. The Execution Command pointing to your brand new gateway_v3
CMD ["uvicorn", "gateway_v3:app", "--host", "0.0.0.0", "--port", "8000"]