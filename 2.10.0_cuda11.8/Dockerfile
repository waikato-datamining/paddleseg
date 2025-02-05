ARG PYTORCH="2.5.1"
ARG CUDA="11.8"
ARG CUDNN="9"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
#FROM ${DOCKER_REGISTRY}ubuntu:22.04

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN pip install -U --no-cache-dir \
        "paddlepaddle-gpu==2.6.2" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4" \
        simple-mask-utils==0.0.1 \
        "simple_palette_utils"

WORKDIR /opt

RUN git clone https://github.com/PaddlePaddle/PaddleSeg.git && \
    cd PaddleSeg && \
    git reset --hard d459390adcec7fa6dd010c21b71aeb73f2afded9 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -v -e . && \
    pip install --no-cache-dir -U "numpy<2.0.0"

COPY bash.bashrc /etc/bash.bashrc
COPY export_config.py /opt/PaddleSeg/tools/
COPY predict*.py /opt/PaddleSeg/tools/
COPY paddleseg_* /usr/bin/

# paddlepaddle-gpu looks for: /usr/local/cuda/lib64/libcudnn.so
RUN ln -s /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/local/cuda/lib64/libcudnn.so

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=/usr/local/cuda-11.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/PaddleSeg
