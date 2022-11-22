# syntax = docker/dockerfile:1.2
FROM gitlab-registry.nrp-nautilus.io/kingb12/docker-dev-env:ibdd
# NOTE: this Dockerfile needs to be build from the repo root directory! e.g. run:
# TODO: replace 'kingb12' with your own username when appropriate!
# docker build -f k8s/Dockerfile -t kingb12/ib-double-descent:latest .
# rather than from this directory:
# docker build -f Dockerfile -t kingb12/ib-double-descent:latest .
# Use the main image shell (which is also supported by conda init)
SHELL ["/usr/bin/zsh", "-c"]

# These can speed up builds that include conda/pip install (like below)
ENV PIP_CACHE_DIR .cache/buildkit/pip
RUN mkdir -p $PIP_CACHE_DIR

# This needs to match the user in the above image, bking2 is the default
ENV USER="ibdd"
ENV PATH="/home/${USER}/miniconda/bin:${PATH}"

USER $USER

# create and enter the project directory
RUN mkdir ib_double_descent
WORKDIR "/home/${USER}/ib_double_descent"

# must run from repo root! Use .dockerignore to filter out things we don't want in the image
COPY --chown=$USER . .

# Set up a conda environment for this project and activate it and install torch and related libraries, then requirements
RUN --mount=type=cache,target=/opt/conda/pkgs conda init zsh && conda create python=3.9 --prefix venv

# conda init is needed before activation, but only takes effect after shell restart. Since each new Dockerfile line
# runs in a new shell instance, this doesn't actually take effect as expected. Instead, we'll change our shell to the
# one installed in the conda environment itself. Make RUN commands use the new environment:
SHELL ["conda", "run", "-p", "./venv", "/bin/bash", "-c"]

RUN --mount=type=cache,target=/opt/conda/pkgs conda install --yes pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
RUN --mount=type=cache,target=/opt/conda/pkgs conda install -c anaconda gxx_linux-64

RUN --mount=type=cache,target=.cache/buildkit/pip pip install pyzmq
RUN --mount=type=cache,target=.cache/buildkit/pip pip install -r requirements.txt
