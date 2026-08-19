FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 spf
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /data && chown spf:spf /data
# Stays root at container start (not USER spf) so the entrypoint can remap
# spf's uid/gid to PUID/PGID and chown /data before dropping to it -- see
# entrypoint.sh.

ENV SPF_DATA_DIR=/data \
    SPF_MEDIA_ROOT=/media

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
