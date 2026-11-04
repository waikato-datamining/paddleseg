ARG PYTORCH="2.14.1"
ARG CUDA="13.0"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1 libglx-mesa0 python3-dev python3-pip python3-venv git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/PaddleSeg.venv

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN /opt/PaddleSeg.venv/bin/pip install -U --no-cache-dir \
        "paddlepaddle-gpu==3.3.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

RUN /opt/PaddleSeg.venv/bin/pip install --no-cache-dir \
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
    rm -Rf .git && \
    /opt/PaddleSeg.venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/PaddleSeg.venv/bin/pip install --no-cache-dir -v -e . && \
    /opt/PaddleSeg.venv/bin/pip install --no-cache-dir -U "numpy<2.0.0" "scipy==1.15.3" && \
    /opt/PaddleSeg.venv/bin/pip install --no-cache-dir -U https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

COPY bash.bashrc /etc/bash.bashrc
COPY export_config.py /opt/PaddleSeg/tools/
COPY predict*.py /opt/PaddleSeg/tools/
COPY paddleseg_* /usr/bin/

ENV PYTHONPATH=/opt/PaddleSeg
