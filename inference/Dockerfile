FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py serve ./
# strip Windows line endings if the file was checked out on Windows, then make it executable
RUN sed -i 's/\r$//' serve && chmod +x serve

# No ENTRYPOINT/CMD on purpose: SageMaker runs `docker run <image> serve`
ENV PATH="/opt/program:${PATH}" PYTHONUNBUFFERED=1
