# hash:sha256:30476850be884d07f629466fb8fa8c97a6dce3bc94a3a5b1a0057525d2520069
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        black==24.4.2 \
        ipykernel==6.29.4 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    anndata==0.10.7 \
    black \
    ipykernel==6.29.4 \
    jupyter==1.0.0 \
    matplotlib==3.8.4 \
    pandas==2.2.2 \
    plotly==5.21.0 \
    pyarrow==16.0.0 \
    scikit-learn==1.4.2 \
    seaborn==0.13.2 \
    umap-learn==0.5.6 \
    xgboost==2.0.3
