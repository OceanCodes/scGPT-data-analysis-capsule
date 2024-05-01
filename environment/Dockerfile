# hash:sha256:9141f2d7d8509628f239d8bade892b2e3df00e8dfda28b5df4c3d7ae5bf871fc
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        ipykernel==6.29.4 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    anndata==0.10.7 \
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

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
