# 1. The Base Blueprint (Lightweight Linux with Python 3.11)
FROM python:3.11-slim

# 2. The Math/C++ Trap Fix (Install system dependencies for FAISS)
# We update the Linux package manager and install the C++ compilers
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set up our working directory inside the Docker image
WORKDIR /app

# 4. Copy the requirements file and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 5. The Machine Learning Pre-bake
# We run a tiny Python command to force the container to download 
# the BGE model directly into the frozen image, saving startup time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 6. Copy all the code into the container
COPY . .

# 7. The Execution Command (What happens when the container starts)
CMD ["uvicorn", "gateway_v2:app", "--host", "0.0.0.0", "--port", "8000"]

