# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set a fixed model cache directory.
ENV TORCH_HOME=/root/.cache/torch

# Install Python and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3.10 python3-pip python3.10-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# update pip and setuptools
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

# install PyTorch with CUDA 12.1 support and other essential packages
# use a dedicated conda env 
RUN conda create --name unsloth_env python=3.10
RUN echo "source activate unsloth_env" > ~/.bashrc
ENV PATH /opt/conda/envs/unsloth_env/bin:$PATH

# as described in the Unsloth.ai Github
RUN conda install -n unsloth_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install matplotlib
RUN pip install --no-deps trl peft accelerate bitsandbytes
RUN pip install autoawq

# install jupiter
RUN pip install jupyter
EXPOSE 8888



#install cmake
RUN apt update
RUN apt install -y cmake

RUN git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 300

# Clone and build llama.cpp
# RUN git config --global http.postBuffer 524288000 && \
# git config --global http.lowSpeedLimit 0 && \
# git config --global http.lowSpeedTime 999999 && \
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
#RUN ls
# # apply fix
# # # RuntimeError: Unsloth: The file 'llama.cpp/llama-quantize' or 
# # 'llama.cpp/quantize' does not exist #748
cd llama.cpp && \
git checkout -f b3345 && \
git submodule update --init --recursive && \
make clean && \
make all -j && \
git log -1

# copy the fine-tuning script into the container
COPY ./unsloth_trainer.py /trainer/unsloth.trainer.py
COPY ./install.sh /trainer/install.sh
COPY ./ollamaunsloth.ipynb /trainer/ollamaunsloth.ipynb



WORKDIR /trainer

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# endless running task to avoid container to be stopped
#CMD [ "/bin/sh" , "-c", "tail -f /dev/null" ]

## docker build -t trainertest .
## docker run -p 8888:8888 --gpus all trainertest