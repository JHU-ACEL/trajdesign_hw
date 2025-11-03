FROM ubuntu:22.04

ARG UNAME
ARG UID
ARG GID

ARG DEBIAN_FRONTEND=noninteractive

COPY ./setup .

RUN apt-get update

RUN apt-get install -y gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev cmake curl apt-utils python3 python3-pip gnupg2 vim git libsm6 libxext6 libxrender-dev nano screen tmux htop openssh-client unzip

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && ln -fs /usr/share/zoneinfo/US/Pacific /etc/localtime && apt-get install -y tzdata && dpkg-reconfigure --frontend noninteractive tzdata

# Install python packages (casadi, jax, etc.)
RUN python3 -m pip install -U pip && python3 -m pip install -U opencv-python numpy Pillow matplotlib scipy sympy matplotlib natsort tqdm h5py seaborn ipython jupyter jupyterlab marimo jax casadi cvxpy qpax pyscipopt

# Install ipopt
RUN apt-get install -y coinor-libipopt-dev -y

# Install acados
RUN /install_acados.sh

ENV ACADOS_SOURCE_DIR="/acados"
ENV ACADOS_PATH="/acados"
ENV LD_LIBRARY_PATH="/acados/lib:${LD_LIBRARY_PATH}"

# Install tera renderer
RUN /install_tera_renderer.sh

CMD ["/bin/bash"]
