ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.11.0+cpu torchvision==0.12.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /opt

RUN git clone https://github.com/PaddlePaddle/PaddleSeg.git && \
    cd PaddleSeg && \
    git reset --hard d459390adcec7fa6dd010c21b71aeb73f2afded9 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -v -e . && \
    pip install --no-cache-dir -U "numpy<2.0.0"

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY export_config.py /opt/PaddleSeg/tools/
COPY predict*.py /opt/PaddleSeg/tools/
COPY paddleseg_* /usr/bin/

# paddlepaddle-gpu looks for: /usr/local/cuda/lib64/libcudnn.so
RUN ln -s /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/local/cuda/lib64/libcudnn.so

ENV PYTHONPATH=/opt/PaddleSeg
