FROM nvidia/cuda:11.1.1-devel-ubuntu18.04
ARG LWLL_SECRET
#ARG LOGGER_REPO_DEPLOY_USER
#ARG LOGGER_REPO_DEPLOY_TOKEN

RUN echo ${LWLL_SECRET}
#RUN echo ${LOGGER_REPO_DEPLOY_USER}
#RUN echo ${LOGGER_REPO_DEPLOY_TOKEN}

#Install other libraries from requirements.txt
RUN apt-get update

#RUN apt-cache search nvidia-driver > result.txt
#RUN sudo docker cp result.txt . 
#CMD ["apt-cache", "search", "nvidia-driver"]

RUN apt-get install -y -q
RUN apt-get install -y build-essential
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get -y install build-essential nghttp2 libnghttp2-dev libssl-dev
RUN apt update && apt-get install -y git
RUN apt-get install dialog apt-utils -y

# --------------Upgrade to Python 3.7------------------
# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# (...)

# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

#-------------------------------------------------

#Copy all files in ~/taglets to /tmp/
COPY . /tmp

RUN cd /tmp && pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN cd /tmp && pip install .
RUN cd /tmp && ./setup.sh
#RUN cd /tmp/ && git clone https://${LOGGER_REPO_DEPLOY_USER}:${LOGGER_REPO_DEPLOY_TOKEN}@gitlab.lollllz.com/brown/logger
#RUN cd /tmp/logger && pip install -e .


RUN useradd --create-home tagletuser
RUN chmod -R 777 /tmp
RUN chmod -R 777 /home/tagletuser/
USER 65534:65534
ENV TORCH_HOME=/tmp

WORKDIR /tmp

CMD bash run_jpl.sh

