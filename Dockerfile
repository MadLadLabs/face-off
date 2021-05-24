# FROM python:3.7
FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6
    # mesa-utils
    # libgl1-mesa-glx

RUN pip3 install \
    pyyaml \
    mediapipe \
    pyvista \
    PyOpenGL \
    PyOpenGL_accelerate

RUN apt-get update && apt-get install -y \
    freeglut3-dev

COPY /src /app

WORKDIR /app

CMD python3 app.py