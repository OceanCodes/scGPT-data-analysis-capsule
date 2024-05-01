# hash:sha256:ba71577284b30c8eeaf1c607577c397095efa5b96d43a1bc57eeb8263cd00336
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        asttokens==2.4.1 \
        ca-certificates==2024.2.2 \
        comm==0.2.2 \
        debugpy==1.8.1 \
        decorator==5.1.1 \
        exceptiongroup==1.2.0 \
        executing==2.0.1 \
        importlib-metadata==7.1.0 \
        importlib_metadata==7.1.0 \
        ipykernel==6.29.3 \
        ipython==8.22.2 \
        jedi==0.19.1 \
        jupyter_client==8.6.1 \
        jupyter_core==5.7.2 \
        libgcc-ng==13.2.0 \
        libgomp==13.2.0 \
        libnsl==2.0.1 \
        libsodium==1.0.18 \
        libsqlite==3.45.3 \
        libstdcxx-ng==13.2.0 \
        libuuid==2.38.1 \
        matplotlib-inline==0.1.7 \
        ncurses==6.4.20240210 \
        nest-asyncio==1.6.0 \
        openssl==3.3.0 \
        packaging==24.0 \
        parso==0.8.4 \
        pexpect==4.9.0 \
        pickleshare==0.7.5 \
        platformdirs==4.2.1 \
        prompt-toolkit==3.0.42 \
        psutil==5.9.8 \
        ptyprocess==0.7.0 \
        pure_eval==0.2.2 \
        pygments==2.17.2 \
        python==3.10.14 \
        python-dateutil==2.9.0.post0 \
        pyzmq==26.0.2 \
        readline==8.2 \
        six==1.16.0 \
        stack_data==0.6.2 \
        tk==8.6.13 \
        tornado==6.4 \
        traitlets==5.14.3 \
        typing_extensions==4.11.0 \
        tzdata==2024.1 \
        wcwidth==0.2.13 \
        zeromq==4.3.5 \
        zipp==3.17.0 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    anndata \
    matplotlib==3.8.4 \
    pandas==2.2.2 \
    plotly==5.21.0 \
    pyarrow==16.0.0 \
    scikit-learn==1.4.2 \
    seaborn==0.13.2 \
    umap-learn \
    xgboost==2.0.3
