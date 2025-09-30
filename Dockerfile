# Usar la imagen de CUDA 11.2 con cuDNN 8.1 y Ubuntu 18.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

# Establecer variables de entorno
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin=${PATH}"

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    build-essential \
    wget \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    tk-dev \
    liblzma-dev \
    xz-utils \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Instalar Python 3.8 (compatible con TensorFlow 2.6 y CUDA 11.2)
ENV PYTHON_VERSION=3.8.12

RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j $(nproc) altinstall && \
    ln -sf /usr/local/bin/python3.8 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.8 /usr/bin/pip3 && \
    rm -rf /usr/src/Python-${PYTHON_VERSION} /usr/src/Python-${PYTHON_VERSION}.tgz

# Instalar TensorFlow 2.6 compatible con CUDA 11.2
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==2.6.0
RUN pip install jupyter notebook ipykernel

# Instalar NumPy 1.19.5 y Pandas 1.2.5 para compatibilidad con TensorFlow 2.6
RUN pip3 install numpy==1.19.5 pandas==1.2.5 matplotlib==3.3.4

# Instalar Keras 2.6 (compatible con TensorFlow 2.6)
RUN pip3 install keras==2.6.0 --no-deps

# Instalar scikit-learn 0.24.2
RUN pip3 install scikit-learn==0.24.2
RUN pip3 install seaborn==0.11.2

# Instalar protobuf compatible con TensorFlow 2.6
RUN pip3 install protobuf==3.19.6
RUN pip install mlxtend

RUN pip install numba==0.56.4

# Verificar instalación
# RUN python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# Comando para ejecutar la aplicación
# CMD ["python3"]
