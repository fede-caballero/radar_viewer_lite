FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Rclone
RUN curl https://rclone.org/install.sh | bash

# Copy and install LROSE
COPY lrose-core-20250105.ubuntu_22.04.amd64.deb /tmp/lrose.deb
RUN apt-get update && apt-get install -y /tmp/lrose.deb && rm /tmp/lrose.deb

# Install Python dependencies
# Cartopy requires libgeos-dev (usually handled by wheel, but good to know)
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    cartopy \
    xarray \
    netCDF4 \
    pyproj \
    scipy \
    requests

WORKDIR /app

# Copy scripts
COPY process_radar.py /app/process_radar.py

CMD ["python3", "/app/process_radar.py"]
