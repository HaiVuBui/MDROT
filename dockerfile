# Start with the nvidia/cuda image with Python and CUDA runtime support
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

WORKDIR /ws

# Install dependencies for Python and Miniconda
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/miniconda/bin:$PATH

# Create a Conda environment with Python 3.8 and install cudatoolkit, numba, and cupy
RUN conda create -n myenv python=3.8 cudatoolkit=11.8 numba cupy -c conda-forge && \
    conda clean -a -y
# Activate the environment and set it as the default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH /opt/miniconda/envs/myenv/bin:$PATH

RUN pip install keras tensorflow
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install scipy 

COPY ./ ./

# Set entrypoint
CMD ["/bin/bash"]

