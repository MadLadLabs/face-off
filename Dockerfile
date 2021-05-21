FROM python:3.7

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    mesa-utils \
    libgl1-mesa-glx

RUN pip3 install \
    pyyaml \
    mediapipe \
    pyvista

COPY /src /app

WORKDIR /app

CMD python3 app.py