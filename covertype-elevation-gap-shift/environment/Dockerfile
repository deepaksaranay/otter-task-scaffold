# TODO before submit: pin the digest -> FROM python:3.10-slim-bookworm@sha256:<digest>
#   (docker manifest inspect python:3.10-slim-bookworm)
FROM python:3.10-slim-bookworm

WORKDIR /app

# TODO before submit: pin the version printed by `apt-cache policy libgomp1` in this image
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY task_inputs/ /app/task_inputs/
RUN mkdir -p /app/outputs

ENV PYTHONPATH=/app
