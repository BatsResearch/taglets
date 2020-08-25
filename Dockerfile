FROM nvidia/cuda:10.1-devel-ubuntu18.04
ARG LWLL_SECRET

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
RUN cd /tmp && pip install .
RUN cd /tmp && ./setup.sh


#RUN addgroup --gid 1000 tagletuser
#RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 tagletuser
RUN useradd --create-home tagletuser
USER tagletuser
#WORKDIR /home/tagletuser

WORKDIR /tmp


#RUN echo ${LWLL_SECRET}

#ENTRYPOINT["taglets.task.jpl.ext_launch"]
CMD ["taglets"]
#CMD ["python","-m","taglets.task.jpl"]


