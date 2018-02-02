FROM debian:stable-slim

MAINTAINER IPP

RUN apt --yes update
RUN apt --yes install --no-install-recommends apt-utils ca-certificates cython gcc git python python-dev python-numpy python-pip python-tables python-yaml python-setuptools

# From https://github.com/docker-library/python/blob/master/Dockerfile-debian.template
ENV LANG C.UTF-8
