ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==2.7.0 torchvision \
    --index-url  https://download.pytorch.org/whl/cpu

WORKDIR /opt

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN pip install -U --no-cache-dir \
        "paddlepaddle==2.6.2" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

RUN git clone https://github.com/PaddlePaddle/PaddleSeg.git && \
    cd PaddleSeg && \
    git reset --hard d459390adcec7fa6dd010c21b71aeb73f2afded9 && \
    rm -Rf .git && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -v -e . && \
    pip install --no-cache-dir -U "numpy<2.0.0"

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4" \
        simple-mask-utils==0.0.1 \
        "simple_palette_utils" \
        "simple_image_ops"

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY export_config.py /opt/PaddleSeg/tools/
COPY predict*.py /opt/PaddleSeg/tools/
COPY paddleseg_* /usr/bin/

ENV PYTHONPATH=/opt/PaddleSeg
