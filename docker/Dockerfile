FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN apt-get update && apt-get upgrade -y
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y install \
    python3 \
    python3-dev \
    python3-distutils \
    python3-pip
RUN pip3 install -U pip wheel setuptools
COPY ./requirements.txt /opt/
WORKDIR /opt/
RUN pip3 install -r requirements.txt
