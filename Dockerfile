FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# Clone the YOLOX repository
COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install Python dependencies
RUN pip3 install -r requirements.txt


WORKDIR /app

# Set the default command to run a terminal
CMD ["bash"]