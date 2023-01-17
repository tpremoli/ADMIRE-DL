    conda env create -f environment.yml
    conda activate training_env

## Make sure [NVIDIA GPU Driver](https://www.nvidia.com/Download/index.aspx) is installed

    nvidia-smi

install conda

    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

Set up library path

    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

install tensorflow

    pip install --upgrade pip
    pip install tensorflow

verify cpu setup (if a tensor is returned, install has been successful)

    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

verify gpu setup (if gpu returned, install has been successful)

    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

